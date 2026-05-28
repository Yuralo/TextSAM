"""Stage 2: multi-query finetuning on ADE20K + LVIS.

Each sample is one image with K text queries (mix of present + absent classes)
and K target masks. We use `TextSAM.forward_multi_query` and a per-query
combined Dice+Focal loss, averaged over queries.

Usage:
    python -m textsam.train.stage2 --config configs/stage2_ade_lvis.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..data import ADE20KDataset, LVISDataset, MergedSegDataset, build_stage2_sampler
from ..models import TextSAM
from ..utils.ckpt import load_checkpoint
from .losses import combined_seg_loss
from .trainer import Trainer


def build_datasets(cfg: dict):
    dcfg = cfg["data"]
    ade_train = ADE20KDataset(
        manifest_path=dcfg["manifest"],
        class_names_file="datasets/ade20k/class_names.txt",
        image_size=dcfg["image_size"],
        split="train",
        queries_per_image=dcfg["queries_per_image"],
        negative_query_ratio=dcfg["negative_query_ratio"],
        augmentations=dcfg.get("augmentations"),
        return_multi=True,
    )
    lvis_train = LVISDataset(
        manifest_path=dcfg["manifest"],
        image_size=dcfg["image_size"],
        split="train",
        queries_per_image=dcfg["queries_per_image"],
        negative_query_ratio=dcfg["negative_query_ratio"],
        augmentations=dcfg.get("augmentations"),
        return_multi=True,
    )
    train = MergedSegDataset([ade_train, lvis_train])

    ade_val = ADE20KDataset(
        manifest_path=dcfg["manifest"],
        class_names_file="datasets/ade20k/class_names.txt",
        image_size=dcfg["image_size"],
        split="val",
        queries_per_image=dcfg["queries_per_image"],
        negative_query_ratio=0.0,    # eval only on present classes
        augmentations=None,
        return_multi=True,
    )
    return train, ade_val


def collate_stage2(items):
    images = torch.stack([x["image"] for x in items])
    masks = torch.stack([x["masks"] for x in items])     # (B, K, H, W)
    texts = [x["texts"] for x in items]                  # list[list[str]]
    return {"image": images, "masks": masks, "texts": texts}


def stage2_step(model: TextSAM, batch: dict, device, amp_dtype, loss_cfg, channels_last=False, loss_size=256):
    images = batch["image"].to(device, non_blocking=True)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    masks = batch["masks"].to(device, non_blocking=True)     # (B, K, H, W)
    texts = batch["texts"]
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
        mask_logits, iou_pred = model.forward_multi_query(images, texts, output_size=loss_size)  # (B,K,1,S,S),(B,K)
        B, K = mask_logits.shape[:2]
        logits_flat = mask_logits.reshape(B * K, *mask_logits.shape[2:])      # (B*K, 1, S, S)
        masks_flat = masks.reshape(B * K, *masks.shape[2:])                   # (B*K, H, W)
        if masks_flat.shape[-1] != logits_flat.shape[-1]:
            masks_flat = torch.nn.functional.interpolate(
                masks_flat.unsqueeze(1), size=logits_flat.shape[-2:], mode="nearest"
            ).squeeze(1)
        iou_flat = iou_pred.reshape(B * K)
        loss, parts = combined_seg_loss(
            logits_flat, masks_flat, iou_flat.unsqueeze(-1),
            dice_weight=loss_cfg["dice_weight"],
            focal_weight=loss_cfg["focal_weight"],
            focal_alpha=loss_cfg["focal_alpha"],
            focal_gamma=loss_cfg["focal_gamma"],
            iou_weight=loss_cfg["iou_weight"],
        )
    return loss, parts, logits_flat, masks_flat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default=None,
                   help="checkpoint to resume from ('auto' = <ckpt_dir>/last.pt). Distinct from warm-start.")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
    # Stage 2 trains at a smaller resolution (512²) to fit K queries per image.
    # Tell the SAM encoder so it interpolates its 1024²-pretrained pos embeddings.
    model_cfg["image_encoder"]["image_size"] = cfg["data"]["image_size"]
    model = TextSAM.from_config(model_cfg)
    print(f"Trainable params: {model.count_trainable_params()/1e6:.2f} M")

    tcfg = cfg["train"]

    # Warm-start adapter+decoder from stage 1 (skipped when --resume continues a stage-2 run).
    resume_from = cfg.get("resume_from")
    if resume_from and Path(resume_from).exists() and not args.resume:
        info = load_checkpoint(resume_from, model, strict=False)
        print(f"Warm-started from {resume_from}: {info['load_msg']}")

    # Stage-2 vocabulary is a fixed, closed set of class names — cache CLIP embeddings.
    if tcfg.get("text_cache", False):
        model.text_encoder.enable_cache(True)
        print("[text-cache] enabled — CLIP text forward runs once per unique class name")

    if tcfg.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
        print("[channels_last] enabled")
    if tcfg.get("compile", False):
        model.adapter = torch.compile(model.adapter, mode="default")
        if tcfg.get("compile_decoder", False):
            model.decoder = torch.compile(model.decoder, mode="default")
        if tcfg.get("compile_encoder", False):
            model.image_encoder.encoder = torch.compile(model.image_encoder.encoder, mode="reduce-overhead")
        print("[compile] enabled")

    train_ds, val_ds = build_datasets(cfg)
    sampler = None
    if cfg["data"].get("sampler", {}).get("use_lvis_rebalance", False):
        sampler = build_stage2_sampler(train_ds.datasets)
    nw = tcfg["num_workers"]
    loader_extras = dict(
        num_workers=nw, pin_memory=True, collate_fn=collate_stage2,
        persistent_workers=bool(tcfg.get("persistent_workers", False)) and nw > 0,
        prefetch_factor=tcfg.get("prefetch_factor", 2) if nw > 0 else None,
    )
    loader_extras = {k: v for k, v in loader_extras.items() if v is not None}
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=(sampler is None),
        sampler=sampler, drop_last=True, **loader_extras,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False, **loader_extras,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[tcfg.get("amp_dtype", "bfloat16")]
    channels_last = tcfg.get("channels_last", False)
    loss_size = int(tcfg.get("loss_size", 256))
    step_fn = lambda batch: stage2_step(model, batch, device, amp_dtype, cfg["loss"],
                                        channels_last=channels_last, loss_size=loss_size)

    trainer = Trainer(model, train_loader, val_loader, cfg, device=device, step_fn=step_fn)

    resume_path = args.resume
    if resume_path == "auto":
        resume_path = str(Path(cfg["logging"]["ckpt_dir"]) / "last.pt")
    if resume_path:
        if Path(resume_path).exists():
            trainer.load(resume_path, resume=True)
        else:
            print(f"[ckpt] --resume given but {resume_path} not found; starting fresh")

    trainer.fit()


if __name__ == "__main__":
    main()
