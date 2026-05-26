"""Stage 1: single-mask text-prompted segmentation on PhraseCut.

Usage:
    python -m textsam.train.stage1 --config configs/stage1_phrasecut.yaml
    python -m textsam.train.stage1 --config configs/stage1_phrasecut.yaml --profile-vram-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..data import PhraseCutDataset
from ..models import TextSAM
from .trainer import Trainer


def build_loaders(cfg: dict):
    dcfg = cfg["data"]
    train_ds = PhraseCutDataset(
        manifest_path=dcfg["manifest"],
        image_size=dcfg["image_size"],
        split="train",
        augmentations=dcfg.get("augmentations"),
    )
    val_ds = PhraseCutDataset(
        manifest_path=dcfg["manifest"],
        image_size=dcfg["image_size"],
        split="val",
        augmentations=None,
    )
    tcfg = cfg["train"]
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg["num_workers"], pin_memory=True, drop_last=True,
        collate_fn=collate_stage1,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=tcfg["num_workers"], pin_memory=True,
        collate_fn=collate_stage1,
    )
    return train_loader, val_loader


def collate_stage1(items):
    images = torch.stack([x["image"] for x in items])
    masks = torch.stack([x["mask"] for x in items])
    texts = [x["text"] for x in items]
    return {"image": images, "mask": masks, "text": texts}


def profile_vram(model, train_loader, device):
    model.train()
    if hasattr(model, "image_encoder"):
        model.image_encoder.eval()
    if hasattr(model, "text_encoder"):
        model.text_encoder.model.eval()
    batch = next(iter(train_loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    texts = batch["text"]
    torch.cuda.reset_peak_memory_stats(device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, iou_pred = model(images, texts)
        loss = ((logits.squeeze(1).sigmoid() - masks) ** 2).mean()
    loss.backward()
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"Peak VRAM: {peak:.2f} GB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--profile-vram-only", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
    model = TextSAM.from_config(model_cfg)
    print(f"Trainable params: {model.count_trainable_params()/1e6:.2f} M")

    train_loader, val_loader = build_loaders(cfg)

    device = args.device if torch.cuda.is_available() else "cpu"

    if args.profile_vram_only:
        profile_vram(model.to(device), train_loader, device)
        return

    trainer = Trainer(model, train_loader, val_loader, cfg, device=device)
    trainer.fit()


if __name__ == "__main__":
    main()
