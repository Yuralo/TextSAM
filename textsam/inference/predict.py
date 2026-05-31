"""CLI + Python API for text-prompted segmentation inference.

Single-word (stage 1 or stage 2):
    python -m textsam.inference.predict \\
        --image examples/dog.jpg --word "dog" \\
        --checkpoint checkpoints/stage2/best.pt --image-size 512 \\
        --out out/dog_mask.png

Multi-word (stage 2 — one image, multiple class names):
    python -m textsam.inference.predict \\
        --image examples/room.jpg \\
        --word "sofa" "lamp" "rug" "window" \\
        --checkpoint checkpoints/stage2/best.pt --image-size 512 \\
        --out out/room

Outputs (single-word):
    <out>.png         binary mask at original image resolution
    <out>_viz.png     RGB overlay
Outputs (multi-word):
    <out>_<word>.png       per-word binary mask
    <out>_panel.png        side-by-side image | colored composite of all masks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from ..data.transforms import SAMPreprocess, to_tensor_chw
from ..models import TextSAM
from ..utils.ckpt import load_checkpoint
from .visualize import overlay_mask


@torch.inference_mode()
def predict_mask(
    model: TextSAM,
    image: Image.Image,
    word: str,
    device: torch.device | str = "cuda",
    threshold: float = 0.0,
    return_logits: bool = False,
) -> np.ndarray:
    """Returns a (H, W) uint8 binary mask at the original image resolution.

    If `return_logits=True`, returns the float mask probabilities instead.
    """
    sam_pre = SAMPreprocess(target_size=model.image_size)
    img_np = np.array(image.convert("RGB"))
    img_t = to_tensor_chw(img_np)
    image_t, _, meta = sam_pre(img_t, None)
    image_t = image_t.unsqueeze(0).to(device)

    model.eval()
    logits, _ = model(image_t, [word])    # (1, 1, S, S)
    # crop pad, then resize back to original
    S = model.image_size
    valid = logits[..., : meta["new_h"], : meta["new_w"]]   # remove zero padding
    prob = torch.sigmoid(valid).float().squeeze(0).squeeze(0)
    prob_up = F.interpolate(
        prob.unsqueeze(0).unsqueeze(0),
        size=(meta["orig_h"], meta["orig_w"]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    if return_logits:
        return prob_up.cpu().numpy()
    mask = (prob_up > 0.5).cpu().numpy().astype(np.uint8) * 255
    return mask


def _palette(K: int) -> np.ndarray:
    import colorsys
    return np.array([
        [int(c * 255) for c in colorsys.hsv_to_rgb((i / K) % 1.0, 0.85, 0.95)]
        for i in range(K)
    ], dtype=np.uint8)


def predict_multi(model, image: Image.Image, words: list[str], device="cuda", threshold: float = 0.0) -> list[np.ndarray]:
    """One image, K text queries -> list of K (H, W) uint8 binary masks at original resolution."""
    sam_pre = SAMPreprocess(target_size=model.image_size)
    img_np = np.array(image.convert("RGB"))
    img_t = to_tensor_chw(img_np)
    image_t, _, meta = sam_pre(img_t, None)
    image_t = image_t.unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        logits, _ = model.forward_multi_query(image_t, [words], output_size=model.image_size)
    # logits: (1, K, 1, S, S)
    valid = logits[0, :, 0, : meta["new_h"], : meta["new_w"]]   # (K, h, w)
    prob = torch.sigmoid(valid).float()
    prob_up = F.interpolate(
        prob.unsqueeze(1),
        size=(meta["orig_h"], meta["orig_w"]),
        mode="bilinear", align_corners=False,
    ).squeeze(1)
    masks = (prob_up > 0.5).cpu().numpy().astype(np.uint8) * 255
    return [m for m in masks]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--word", nargs="+", required=True,
                   help="one or more text queries; if more than one, runs the multi-query path")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--image-size", type=int, default=None,
                   help="override the encoder input size (use 512 for stage-2 ckpts, 1024 for stage-1)")
    p.add_argument("--out", default="out/mask")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model_cfg = yaml.safe_load(Path(args.model_config).read_text())
    if args.image_size is not None:
        model_cfg["image_encoder"]["image_size"] = args.image_size
    model = TextSAM.from_config(model_cfg).to(device)
    info = load_checkpoint(args.checkpoint, model, strict=False)
    print(f"loaded {args.checkpoint}: {info['load_msg']}")
    print(f"model.image_size = {model.image_size}")

    image = Image.open(args.image)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- single-word path (stage 1 or stage 2 with one query) ----
    if len(args.word) == 1:
        mask = predict_mask(model, image, args.word[0], device=device)
        out_png = out_path.with_suffix(".png")
        Image.fromarray(mask).save(out_png)
        viz_path = out_path.with_name(out_path.stem + "_viz.png")
        overlay_mask(image, mask // 255).save(viz_path)
        print(f"mask: {out_png}")
        print(f"viz:  {viz_path}")
        return

    # ---- multi-word path (stage 2 multi-query) ----
    masks = predict_multi(model, image, args.word, device=device)
    img_np = np.array(image.convert("RGB"))
    palette = _palette(len(args.word))
    composite = img_np.astype(np.float32).copy()
    for k, (w, m) in enumerate(zip(args.word, masks)):
        m_bin = (m > 127)
        # save per-word mask
        Image.fromarray(m).save(out_path.with_name(f"{out_path.stem}_{w.replace(' ', '_')}.png"))
        # blend into composite
        composite[m_bin] = composite[m_bin] * 0.5 + palette[k].astype(np.float32) * 0.5

    composite = composite.clip(0, 255).astype(np.uint8)

    # side-by-side panel: image | composite, with class-color legend printed
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_np); axes[0].set_title("input"); axes[0].axis("off")
        axes[1].imshow(composite); axes[1].set_title("predictions"); axes[1].axis("off")
        legend = "  ".join(
            f"■ {w}" for w in args.word
        )
        # color legend below
        for k, w in enumerate(args.word):
            fig.text(0.05 + k * 0.18, 0.02, f"■ {w}", color=np.array(palette[k]) / 255.0,
                     fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        panel_path = out_path.with_name(f"{out_path.stem}_panel.png")
        plt.savefig(panel_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"panel: {panel_path}")
    except ImportError:
        # fall back: save the composite directly
        Image.fromarray(composite).save(out_path.with_name(f"{out_path.stem}_composite.png"))

    print(f"per-word masks saved next to {out_path}")


if __name__ == "__main__":
    main()
