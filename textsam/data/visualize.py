"""Dataset visualization for PhraseCut (stage-1 manifest).

Renders three views to disk under `--output`:

  - samples_train.png / samples_val.png : grids of (image + mask overlay + phrase)
  - transforms.png                      : raw vs. augmented vs. SAM-preprocessed
                                          rows for the same entries
  - stats.png                           : phrase-length, phrases-per-image,
                                          top phrases, dataset summary

Usage:
    python -m textsam.data.visualize \\
        --manifest datasets/manifest_stage1.jsonl \\
        --output viz/ \\
        --samples-n 16
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .phrasecut import _decode_rle, _rasterize_polygons
from .transforms import SAM_PIXEL_MEAN, SAM_PIXEL_STD, SAMPreprocess, build_joint_transform, to_tensor_chw


# -------------------- helpers --------------------

def _denorm_sam(image_t: torch.Tensor) -> np.ndarray:
    """SAM-normalized (3, H, W) float -> HWC uint8 RGB for display."""
    img = image_t.detach().cpu().float()
    img = img * SAM_PIXEL_STD + SAM_PIXEL_MEAN          # back to [0, 255]
    img = img.clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return img


def _overlay_mask(img: np.ndarray, mask: np.ndarray, color=(255, 48, 48), alpha=0.45) -> np.ndarray:
    """Alpha-blend a binary mask onto an RGB uint8 image."""
    if mask.dtype != np.bool_:
        mask = mask > 0.5
    out = img.copy()
    color_arr = np.array(color, dtype=np.uint8)
    out[mask] = (out[mask].astype(np.float32) * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return out


def _load_entries(manifest_path: str | Path) -> list[dict]:
    entries = []
    with open(manifest_path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("dataset") == "phrasecut":
                entries.append(e)
    return entries


def _load_raw(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load the on-disk image + the GT binary mask for one entry."""
    img = np.array(Image.open(entry["image"]).convert("RGB"))
    H, W = img.shape[:2]
    if "rle" in entry:
        mask = _decode_rle(entry["rle"])
    else:
        mask = _rasterize_polygons(entry["polygons"], H, W)
    return img, mask


# -------------------- views --------------------

def visualize_samples(entries: list[dict], n: int, out_path: Path, title: str, seed: int = 0):
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    picks = rng.sample(entries, min(n, len(entries)))
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.atleast_2d(axes).flatten()

    for ax, e in zip(axes, picks):
        img, mask = _load_raw(e)
        ax.imshow(_overlay_mask(img, mask))
        ax.set_title(f'"{e["text"]}"', fontsize=10)
        ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved {out_path}")


def visualize_transforms(entries: list[dict], n: int, image_size: int, out_path: Path, seed: int = 0):
    """3 columns per row: raw / joint-augmented / SAM-preprocessed (denormalized)."""
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    picks = rng.sample(entries, min(n, len(entries)))
    aug_cfg = {"flip_horizontal": 0.5, "color_jitter": 0.2, "random_resized_crop": [0.7, 1.0]}
    joint = build_joint_transform(image_size, aug_cfg)
    sam_pre = SAMPreprocess(target_size=image_size)

    fig, axes = plt.subplots(len(picks), 3, figsize=(12, 4 * len(picks)))
    if len(picks) == 1:
        axes = np.array([axes])

    for i, e in enumerate(picks):
        raw_img, raw_mask = _load_raw(e)

        aug_img, aug_mask = joint(raw_img, raw_mask)

        sam_img_t, sam_mask_t, _ = sam_pre(to_tensor_chw(aug_img), torch.from_numpy(aug_mask).float())
        sam_img = _denorm_sam(sam_img_t)
        sam_mask = sam_mask_t.cpu().numpy()

        axes[i][0].imshow(_overlay_mask(raw_img, raw_mask))
        axes[i][0].set_title(f'raw  ({raw_img.shape[1]}×{raw_img.shape[0]})\n"{e["text"]}"', fontsize=10)
        axes[i][0].axis("off")

        axes[i][1].imshow(_overlay_mask(aug_img, aug_mask))
        axes[i][1].set_title("after joint augment\n(flip · color-jitter · random-resized-crop)", fontsize=10)
        axes[i][1].axis("off")

        axes[i][2].imshow(_overlay_mask(sam_img, sam_mask))
        axes[i][2].set_title(f"after SAM preprocess\n({image_size}×{image_size}, normalized → denorm view)", fontsize=10)
        axes[i][2].axis("off")

    fig.suptitle("Per-sample transform pipeline", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved {out_path}")


def visualize_stats(entries: list[dict], out_path: Path):
    import matplotlib.pyplot as plt

    by_split = {"train": [], "val": []}
    for e in entries:
        sp = e.get("split")
        if sp in by_split:
            by_split[sp].append(e)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) phrase length distribution
    lens_train = [len(e["text"].split()) for e in by_split["train"]]
    lens_val = [len(e["text"].split()) for e in by_split["val"]]
    bins = range(1, 21)
    axes[0, 0].hist([lens_train, lens_val], bins=bins, label=["train", "val"],
                    alpha=0.75, color=["#3b82f6", "#f59e0b"])
    axes[0, 0].set_xlabel("phrase length (words)")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_title("Phrase length distribution")
    axes[0, 0].legend()

    # (0,1) phrases per image (train)
    img_counts = Counter(e["image"] for e in by_split["train"])
    ppi = list(img_counts.values())
    axes[0, 1].hist(ppi, bins=range(1, max(ppi) + 2), color="#10b981")
    axes[0, 1].set_xlabel("phrases per image (train)")
    axes[0, 1].set_ylabel("image count")
    axes[0, 1].set_title(f"Phrases per image — mean {np.mean(ppi):.1f}, median {int(np.median(ppi))}")

    # (1,0) top phrases
    top = Counter(e["text"].lower().strip() for e in by_split["train"]).most_common(20)
    labels, freqs = zip(*top[::-1])
    axes[1, 0].barh(labels, freqs, color="#6366f1")
    axes[1, 0].set_xlabel("frequency (train)")
    axes[1, 0].set_title("Top 20 phrases")

    # (1,1) summary panel
    n_train = len(by_split["train"])
    n_val = len(by_split["val"])
    uniq_train_imgs = len(img_counts)
    uniq_val_imgs = len({e["image"] for e in by_split["val"]})
    uniq_phrases = len({e["text"].lower() for e in by_split["train"]})
    summary = [
        f"(image, phrase) pairs",
        f"  train: {n_train:>10,}",
        f"  val:   {n_val:>10,}",
        f"",
        f"unique images",
        f"  train: {uniq_train_imgs:>10,}",
        f"  val:   {uniq_val_imgs:>10,}",
        f"",
        f"unique phrases (train): {uniq_phrases:,}",
        f"avg phrases / image:    {np.mean(ppi):.2f}",
        f"max phrases / image:    {max(ppi)}",
        f"median phrase length:   {int(np.median(lens_train))} words",
    ]
    axes[1, 1].axis("off")
    axes[1, 1].text(0.0, 0.95, "\n".join(summary),
                    fontsize=12, family="monospace", va="top", ha="left")
    axes[1, 1].set_title("Dataset summary")

    fig.suptitle("PhraseCut stage-1 manifest — statistics", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved {out_path}")


# -------------------- entry point --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="datasets/manifest_stage1.jsonl")
    p.add_argument("--output", default="viz")
    p.add_argument("--samples-n", type=int, default=16, help="how many samples per split grid")
    p.add_argument("--transforms-n", type=int, default=4, help="how many rows in the transform comparison")
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-samples", action="store_true")
    p.add_argument("--skip-transforms", action="store_true")
    p.add_argument("--skip-stats", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = _load_entries(args.manifest)
    train = [e for e in entries if e.get("split") == "train"]
    val = [e for e in entries if e.get("split") == "val"]
    print(f"[viz] manifest {args.manifest}: {len(train):,} train, {len(val):,} val entries")

    if not args.skip_samples:
        visualize_samples(train, args.samples_n, out_dir / "samples_train.png",
                          title="PhraseCut train — random samples (mask + phrase)", seed=args.seed)
        visualize_samples(val, min(args.samples_n, 8), out_dir / "samples_val.png",
                          title="PhraseCut val — random samples (mask + phrase)", seed=args.seed)

    if not args.skip_transforms:
        visualize_transforms(train, args.transforms_n, args.image_size,
                             out_dir / "transforms.png", seed=args.seed)

    if not args.skip_stats:
        visualize_stats(entries, out_dir / "stats.png")


if __name__ == "__main__":
    main()
