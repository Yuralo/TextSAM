"""Joint image+mask augmentation and SAM-style preprocessing.

SAM expects images normalized with `pixel_mean=[123.675,116.28,103.53]`,
`pixel_std=[58.395,57.12,57.375]` and resized so the longest side is 1024,
then padded to 1024x1024 with zeros (bottom/right).

For training we apply random horizontal flip + color jitter + random-resized-crop
*jointly* to image and mask via albumentations, then run SAMPreprocess.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
SAM_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)


class SAMPreprocess:
    """Resize longest side to `target_size`, pad to square, normalize.

    Operates on torch tensors:
        image: (3, H, W) uint8 or float in [0,255]
        mask:  (H, W)    uint8/float (binary or class-id)

    Returns:
        image: (3, target_size, target_size) float32, SAM-normalized
        mask:  (target_size, target_size)     float32, same scale as input
        meta:  dict with original size and pad
    """

    def __init__(self, target_size: int = 1024):
        self.target_size = target_size

    def __call__(self, image: Tensor, mask: Tensor | None = None) -> Tuple[Tensor, Tensor | None, dict]:
        if image.dtype != torch.float32:
            image = image.float()
        c, h, w = image.shape
        scale = self.target_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        pad_h, pad_w = self.target_size - new_h, self.target_size - new_w
        image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
        image = (image - SAM_PIXEL_MEAN.to(image.dtype)) / SAM_PIXEL_STD.to(image.dtype)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()
            mask = F.interpolate(mask.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
            mask = F.pad(mask, (0, pad_w, 0, pad_h), value=0.0)
            mask = mask.squeeze(0)

        meta = {"orig_h": h, "orig_w": w, "new_h": new_h, "new_w": new_w,
                "pad_h": pad_h, "pad_w": pad_w, "scale": scale}
        return image, mask, meta


def build_joint_transform(image_size: int, augs: dict | None = None):
    """Returns a callable (image_np HWC uint8, mask_np HW uint8) -> (image_np, mask_np)
    that applies augmentation. We use albumentations for image+mask sync.
    """
    try:
        import albumentations as A
    except ImportError:
        raise ImportError("pip install albumentations")

    augs = augs or {}
    transforms = []
    rrc = augs.get("random_resized_crop")
    if rrc:
        transforms.append(A.RandomResizedCrop(size=(image_size, image_size), scale=tuple(rrc), p=1.0))
    else:
        transforms.append(A.LongestMaxSize(max_size=image_size))
        transforms.append(A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0))

    if augs.get("flip_horizontal", 0) > 0:
        transforms.append(A.HorizontalFlip(p=augs["flip_horizontal"]))

    cj = augs.get("color_jitter", 0)
    if cj > 0:
        transforms.append(A.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=cj / 2, p=0.8))

    pipe = A.Compose(transforms)

    def apply(image: np.ndarray, mask: np.ndarray):
        out = pipe(image=image, mask=mask)
        return out["image"], out["mask"]

    return apply


def to_tensor_chw(image_np: np.ndarray) -> Tensor:
    """HWC uint8 -> CHW float [0,255]."""
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    t = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float()
    return t
