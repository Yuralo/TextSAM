"""CLI + Python API for text-prompted segmentation inference.

Usage:
    python -m textsam.inference.predict \\
        --image  examples/dog.jpg \\
        --word   "dog" \\
        --checkpoint checkpoints/stage2/best.pt \\
        --model-config configs/model.yaml \\
        --out    out/dog_mask.png

Outputs:
    <out>.png         binary mask at the original image resolution (uint8 0/255)
    <out>_viz.png     RGB overlay visualisation
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--word", required=True, help="text query (a word or short phrase)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--out", default="out/mask.png")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model_cfg = yaml.safe_load(Path(args.model_config).read_text())
    model = TextSAM.from_config(model_cfg).to(device)
    info = load_checkpoint(args.checkpoint, model, strict=False)
    print(f"loaded {args.checkpoint}: {info['load_msg']}")

    image = Image.open(args.image)
    mask = predict_mask(model, image, args.word, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)
    viz = overlay_mask(image, mask // 255)
    viz_path = out_path.with_name(out_path.stem + "_viz.png")
    viz.save(viz_path)
    print(f"mask: {out_path}")
    print(f"viz:  {viz_path}")


if __name__ == "__main__":
    main()
