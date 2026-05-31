"""Evaluate a stage-2 TextSAM checkpoint on the ADE20K val split.

Produces, under `--output`:
  - metrics.json     overall mIoU, IoU@{0.5,0.7,0.9}, per-class mIoU (top/bottom 20)
  - iou_hist.png     per-query IoU distribution
  - qualitative.png  image | GT composite | prediction composite, with class names + IoUs

Usage:
    python -m textsam.inference.evaluate_stage2 \\
        --checkpoint checkpoints/stage2/best.pt \\
        --model-config configs/model.yaml \\
        --stage2-config configs/stage2_ade_lvis.yaml \\
        --output eval/stage2
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.ade20k import ADE20KDataset
from ..data.transforms import SAM_PIXEL_MEAN, SAM_PIXEL_STD
from ..models import TextSAM
from ..utils.ckpt import load_checkpoint
from ..utils.metrics import mask_iou


def _collate(items):
    return {
        "image": torch.stack([x["image"] for x in items]),
        "masks": torch.stack([x["masks"] for x in items]),     # (B, K, H, W)
        "texts": [x["texts"] for x in items],                  # list[list[str]]
    }


def _denorm(image_t: torch.Tensor) -> np.ndarray:
    img = image_t.detach().cpu().float()
    img = img * SAM_PIXEL_STD + SAM_PIXEL_MEAN
    return img.clamp(0, 255).byte().permute(1, 2, 0).numpy()


# K visually-distinct colors (HSV ring -> RGB)
def _palette(K: int) -> np.ndarray:
    import colorsys
    out = []
    for i in range(K):
        h = (i / K) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        out.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(out, dtype=np.uint8)


def _composite(img: np.ndarray, masks: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay K binary masks (K,H,W) on an HxWx3 uint8 image with distinct colors."""
    out = img.astype(np.float32).copy()
    colors = _palette(masks.shape[0])
    for k in range(masks.shape[0]):
        m = masks[k].astype(bool)
        if not m.any():
            continue
        c = colors[k].astype(np.float32)
        out[m] = out[m] * (1 - alpha) + c * alpha
    return out.clip(0, 255).astype(np.uint8)


@torch.inference_mode()
def run_eval(model, loader, device, output_size):
    """Returns lists: per-query iou, class-name per query, image index per query."""
    model.eval()
    ious: list[float] = []
    classes: list[str] = []
    img_indices: list[int] = []
    sample_idx = 0
    for batch in tqdm(loader, desc="eval"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)         # (B, K, H, W)
        texts = batch["texts"]
        logits, _ = model.forward_multi_query(images, texts, output_size=output_size)  # (B,K,1,S,S)
        B, K = logits.shape[:2]
        logits_flat = logits.reshape(B * K, *logits.shape[2:])        # (B*K, 1, S, S)
        masks_flat = masks.reshape(B * K, *masks.shape[2:])           # (B*K, H, W)
        if masks_flat.shape[-1] != logits_flat.shape[-1]:
            masks_flat = torch.nn.functional.interpolate(
                masks_flat.unsqueeze(1), size=logits_flat.shape[-2:], mode="nearest"
            ).squeeze(1)
        per = mask_iou(logits_flat.float(), masks_flat).tolist()       # (B*K,)
        ious.extend(per)
        for b in range(B):
            for k in range(K):
                classes.append(texts[b][k])
                img_indices.append(sample_idx + b)
        sample_idx += B
    return ious, classes, img_indices


def aggregate(ious, classes):
    arr = np.array(ious)
    by_class = defaultdict(list)
    for iou, c in zip(ious, classes):
        by_class[c].append(iou)
    per_class = {c: float(np.mean(v)) for c, v in by_class.items()}
    sorted_classes = sorted(per_class.items(), key=lambda kv: kv[1])
    return {
        "n_queries": int(arr.size),
        "n_classes_seen": len(per_class),
        "miou": float(arr.mean()),
        "median_iou": float(np.median(arr)),
        "iou@0.5": float((arr >= 0.5).mean()),
        "iou@0.7": float((arr >= 0.7).mean()),
        "iou@0.9": float((arr >= 0.9).mean()),
        "bottom_20_classes": {c: round(v, 4) for c, v in sorted_classes[:20]},
        "top_20_classes": {c: round(v, 4) for c, v in sorted_classes[-20:][::-1]},
    }


def plot_hist(ious, out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ious, bins=np.linspace(0, 1, 21), color="#3b82f6", edgecolor="white")
    ax.axvline(np.mean(ious), color="#ef4444", linestyle="--", label=f"mean {np.mean(ious):.3f}")
    ax.axvline(np.median(ious), color="#10b981", linestyle="--", label=f"median {np.median(ious):.3f}")
    ax.set_xlabel("per-query IoU")
    ax.set_ylabel("count")
    ax.set_title("ADE20K val (stage 2) — IoU distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] saved {out_path}")


@torch.inference_mode()
def plot_qualitative(model, dataset, ious, img_indices, device, output_size, out_path: Path, k_rows: int = 3, seed: int = 0):
    """Group by image; pick best / worst / random images by average IoU across that image's K queries."""
    import matplotlib.pyplot as plt

    by_img: dict[int, list[float]] = defaultdict(list)
    for iou, idx in zip(ious, img_indices):
        by_img[idx].append(iou)
    img_avg = {i: float(np.mean(v)) for i, v in by_img.items()}
    sorted_imgs = sorted(img_avg.items(), key=lambda kv: kv[1])
    worst = [i for i, _ in sorted_imgs[:k_rows]]
    best = [i for i, _ in sorted_imgs[::-1][:k_rows]]
    rng = random.Random(seed)
    rand = rng.sample(list(img_avg.keys()), min(k_rows, len(img_avg)))
    groups = [("best", best), ("worst", worst), ("random", rand)]

    total = sum(len(g) for _, g in groups)
    fig, axes = plt.subplots(total, 3, figsize=(13, 4.2 * total))
    if total == 1:
        axes = np.array([axes])

    model.eval()
    row = 0
    for label, idxs in groups:
        for j, dataset_idx in enumerate(idxs):
            item = dataset[dataset_idx]
            image_t = item["image"].unsqueeze(0).to(device)
            masks_gt = item["masks"].cpu().numpy()        # (K, H, W)
            texts = [item["texts"]]
            logits, _ = model.forward_multi_query(image_t, texts, output_size=output_size)
            preds = (torch.sigmoid(logits.squeeze(0).squeeze(1)) > 0.5).cpu().numpy()  # (K,S,S)

            # match resolutions for display (use model output size as canvas)
            if masks_gt.shape[-1] != preds.shape[-1]:
                from PIL import Image as PILImage
                gt_rs = []
                for k in range(masks_gt.shape[0]):
                    m = PILImage.fromarray(masks_gt[k].astype(np.uint8) * 255).resize(
                        (preds.shape[-1], preds.shape[-2]), resample=PILImage.NEAREST
                    )
                    gt_rs.append((np.array(m) > 127).astype(np.uint8))
                masks_gt = np.stack(gt_rs, 0)

            img_np = _denorm(item["image"])  # (S, S, 3) at sam-preprocessed size
            if img_np.shape[0] != preds.shape[-1]:
                from PIL import Image as PILImage
                img_np = np.array(PILImage.fromarray(img_np).resize((preds.shape[-1], preds.shape[-2]), PILImage.BILINEAR))

            gt_comp = _composite(img_np, masks_gt)
            pr_comp = _composite(img_np, preds)

            # legend text: class names + IoU
            per_q = []
            for k_idx, name in enumerate(item["texts"]):
                inter = np.logical_and(preds[k_idx] > 0, masks_gt[k_idx] > 0).sum()
                union = np.logical_or(preds[k_idx] > 0, masks_gt[k_idx] > 0).sum()
                iou = inter / max(union, 1)
                per_q.append(f"{name}={iou:.2f}")
            legend = " · ".join(per_q)

            axes[row][0].imshow(img_np)
            tag = f"[{label}]" if j == 0 else ""
            axes[row][0].set_title(f"{tag} image · avg IoU {img_avg[dataset_idx]:.2f}", fontsize=10)
            axes[row][0].axis("off")

            axes[row][1].imshow(gt_comp)
            axes[row][1].set_title("GT (color = class)", fontsize=10)
            axes[row][1].axis("off")

            axes[row][2].imshow(pr_comp)
            axes[row][2].set_title("prediction", fontsize=10)
            axes[row][2].axis("off")

            # legend underneath the row
            fig.text(0.5, 1 - (row + 0.95) / total, legend, ha="center", fontsize=8)
            row += 1

    fig.suptitle("TextSAM stage-2 — qualitative results (ADE20K val)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--stage2-config", default="configs/stage2_ade_lvis.yaml")
    p.add_argument("--output", default="eval/stage2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--queries-per-image", type=int, default=8)
    p.add_argument("--negative-query-ratio", type=float, default=0.0,
                   help="fraction of queries that are *absent* classes (refusal test)")
    p.add_argument("--qualitative-k", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)

    s2cfg = yaml.safe_load(Path(args.stage2_config).read_text())
    image_size = int(s2cfg["data"]["image_size"])

    device = args.device if torch.cuda.is_available() else "cpu"
    model_cfg = yaml.safe_load(Path(args.model_config).read_text())
    model_cfg["image_encoder"]["image_size"] = image_size   # match stage 2 (e.g. 512)
    model = TextSAM.from_config(model_cfg).to(device)
    info = load_checkpoint(args.checkpoint, model, strict=False)
    print(f"[eval] loaded {args.checkpoint}: {info['load_msg']}")

    val_ds = ADE20KDataset(
        manifest_path=s2cfg["data"]["manifest"],
        class_names_file="datasets/ade20k/class_names.txt",
        image_size=image_size,
        split="val",
        queries_per_image=args.queries_per_image,
        negative_query_ratio=args.negative_query_ratio,
        augmentations=None,
        return_multi=True,
    )
    if args.limit:
        val_ds.entries = val_ds.entries[: args.limit]
    print(f"[eval] {len(val_ds)} val images, K={args.queries_per_image} queries/image"
          f" ({args.negative_query_ratio:.0%} absent)")

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=_collate)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ious, classes, img_idx = run_eval(model, loader, device, output_size=image_size)
    metrics = aggregate(ious, classes)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[eval] metrics:")
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))
    print(f"[eval] top-20 / bottom-20 classes written to metrics.json")

    plot_hist(ious, out_dir / "iou_hist.png")
    plot_qualitative(model, val_ds, ious, img_idx, device, image_size,
                     out_dir / "qualitative.png", k_rows=args.qualitative_k, seed=args.seed)


if __name__ == "__main__":
    main()
