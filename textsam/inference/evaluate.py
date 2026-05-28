"""Evaluate a trained TextSAM checkpoint on the PhraseCut val split.

Produces, under `--output`:
  - metrics.json        aggregate mIoU + IoU@{0.5,0.7,0.9} + per-phrase-length breakdown
  - iou_hist.png        histogram of per-sample IoU
  - qualitative.png     best / worst / random predictions: image | GT | prediction

Usage:
    python -m textsam.inference.evaluate \\
        --checkpoint checkpoints/stage1/best.pt \\
        --model-config configs/model.yaml \\
        --manifest datasets/manifest_stage1.jsonl \\
        --output eval/stage1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.phrasecut import PhraseCutDataset, _decode_rle, _rasterize_polygons
from ..data.visualize import _load_raw, _overlay_mask
from ..models import TextSAM
from ..utils.ckpt import load_checkpoint
from ..utils.metrics import mask_iou
from .predict import predict_mask


def _collate(items):
    return {
        "image": torch.stack([x["image"] for x in items]),
        "mask": torch.stack([x["mask"] for x in items]),
        "text": [x["text"] for x in items],
    }


@torch.inference_mode()
def run_quantitative(model, loader, device, output_size) -> list[float]:
    """Per-sample IoU across the whole val loader (order matches dataset)."""
    model.eval()
    ious: list[float] = []
    for batch in tqdm(loader, desc="eval"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits, _ = model(images, batch["text"], output_size=output_size)
        if masks.shape[-1] != logits.shape[-1]:
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1), size=logits.shape[-2:], mode="nearest"
            ).squeeze(1)
        ious.extend(mask_iou(logits.float(), masks).tolist())
    return ious


def aggregate(ious: list[float], entries: list[dict]) -> dict:
    arr = np.array(ious)
    by_len: dict[int, list[float]] = {}
    for iou, e in zip(ious, entries):
        L = min(len(e["text"].split()), 6)  # bucket 6+ together
        by_len.setdefault(L, []).append(iou)
    return {
        "n": int(arr.size),
        "miou": float(arr.mean()),
        "median_iou": float(np.median(arr)),
        "iou@0.5": float((arr >= 0.5).mean()),
        "iou@0.7": float((arr >= 0.7).mean()),
        "iou@0.9": float((arr >= 0.9).mean()),
        "miou_by_phrase_len": {str(k): round(float(np.mean(v)), 4) for k, v in sorted(by_len.items())},
    }


def plot_hist(ious: list[float], out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ious, bins=np.linspace(0, 1, 21), color="#3b82f6", edgecolor="white")
    ax.axvline(np.mean(ious), color="#ef4444", linestyle="--", label=f"mean {np.mean(ious):.3f}")
    ax.axvline(np.median(ious), color="#10b981", linestyle="--", label=f"median {np.median(ious):.3f}")
    ax.set_xlabel("per-sample IoU")
    ax.set_ylabel("count")
    ax.set_title("PhraseCut val — IoU distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] saved {out_path}")


def _gt_mask_raw(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    img, mask = _load_raw(entry)
    return img, mask


def plot_qualitative(model, entries, ious, device, out_path: Path, k: int = 4, seed: int = 0):
    """Rows grouped: best-k, worst-k, random-k. Cols: image | GT | prediction."""
    import matplotlib.pyplot as plt

    order = np.argsort(ious)
    worst = order[:k].tolist()
    best = order[::-1][:k].tolist()
    rng = random.Random(seed)
    rand = rng.sample(range(len(entries)), min(k, len(entries)))
    groups = [("best", best), ("worst", worst), ("random", rand)]

    total_rows = sum(len(idxs) for _, idxs in groups)
    fig, axes = plt.subplots(total_rows, 3, figsize=(12, 4 * total_rows))
    if total_rows == 1:
        axes = np.array([axes])

    row = 0
    for label, idxs in groups:
        for j, idx in enumerate(idxs):
            e = entries[idx]
            img, gt = _gt_mask_raw(e)
            pred = predict_mask(model, Image.fromarray(img), e["text"], device=device)  # (H,W) 0/255
            pred_bin = (pred > 127).astype(np.uint8)

            inter = np.logical_and(pred_bin, gt > 0).sum()
            union = np.logical_or(pred_bin, gt > 0).sum()
            iou = inter / max(union, 1)

            axes[row][0].imshow(img)
            tag = f"[{label}]" if j == 0 else ""
            axes[row][0].set_title(f'{tag} "{e["text"]}"', fontsize=10)
            axes[row][0].axis("off")

            axes[row][1].imshow(_overlay_mask(img, gt, color=(48, 220, 48)))
            axes[row][1].set_title("ground truth", fontsize=10)
            axes[row][1].axis("off")

            axes[row][2].imshow(_overlay_mask(img, pred_bin, color=(255, 48, 48)))
            axes[row][2].set_title(f"prediction · IoU={iou:.3f}", fontsize=10)
            axes[row][2].axis("off")
            row += 1

    fig.suptitle("TextSAM stage-1 — qualitative results (green=GT, red=pred)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--manifest", default="datasets/manifest_stage1.jsonl")
    p.add_argument("--output", default="eval/stage1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--eval-size", type=int, default=1024,
                   help="resolution at which IoU is computed (1024 = full; 256 = match train loss)")
    p.add_argument("--qualitative-k", type=int, default=4, help="rows per group (best/worst/random)")
    p.add_argument("--limit", type=int, default=None, help="cap number of val samples (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model_cfg = yaml.safe_load(Path(args.model_config).read_text())
    model = TextSAM.from_config(model_cfg).to(device)
    info = load_checkpoint(args.checkpoint, model, strict=False)
    print(f"[eval] loaded {args.checkpoint}: {info['load_msg']}")

    val_ds = PhraseCutDataset(manifest_path=args.manifest, image_size=model.image_size,
                              split="val", augmentations=None)
    entries = list(val_ds.entries)
    if args.limit:
        val_ds.entries = entries = entries[: args.limit]
    print(f"[eval] {len(entries)} val samples")

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=_collate)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ious = run_quantitative(model, loader, device, args.eval_size)
    metrics = aggregate(ious, entries)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[eval] metrics:")
    print(json.dumps(metrics, indent=2))

    plot_hist(ious, out_dir / "iou_hist.png")
    plot_qualitative(model, entries, ious, device, out_dir / "qualitative.png",
                     k=args.qualitative_k, seed=args.seed)


if __name__ == "__main__":
    main()
