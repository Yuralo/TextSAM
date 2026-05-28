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


def stage2_step(model: TextSAM, batch: dict, device, amp_dtype, loss_cfg):
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["masks"].to(device, non_blocking=True)     # (B, K, H, W)
    texts = batch["texts"]
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
        mask_logits, iou_pred = model.forward_multi_query(images, texts)  # (B,K,1,H,W), (B,K)
        B, K = mask_logits.shape[:2]
        # flatten B*K for loss
        logits_flat = mask_logits.reshape(B * K, *mask_logits.shape[2:])
        masks_flat = masks.reshape(B * K, *masks.shape[2:])
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
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
    # Stage 2 trains at a smaller resolution (512²) to fit K queries per image.
    # Tell the SAM encoder so it interpolates its 1024²-pretrained pos embeddings.
    model_cfg["image_encoder"]["image_size"] = cfg["data"]["image_size"]
    model = TextSAM.from_config(model_cfg)
    print(f"Trainable params: {model.count_trainable_params()/1e6:.2f} M")

    resume = cfg.get("resume_from")
    if resume and Path(resume).exists():
        info = load_checkpoint(resume, model, strict=False)
        print(f"Warm-started from {resume}: {info['load_msg']}")

    train_ds, val_ds = build_datasets(cfg)
    tcfg = cfg["train"]
    sampler = None
    if cfg["data"].get("sampler", {}).get("use_lvis_rebalance", False):
        sampler = build_stage2_sampler(train_ds.datasets)
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=(sampler is None),
        sampler=sampler, num_workers=tcfg["num_workers"], pin_memory=True, drop_last=True,
        collate_fn=collate_stage2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=tcfg["num_workers"], pin_memory=True,
        collate_fn=collate_stage2,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[tcfg.get("amp_dtype", "bfloat16")]
    step_fn = lambda batch: stage2_step(model, batch, device, amp_dtype, cfg["loss"])

    trainer = Trainer(model, train_loader, val_loader, cfg, device=device, step_fn=step_fn)
    trainer.fit()


if __name__ == "__main__":
    main()
