"""Mask visualization: colored overlay + contour outline."""

from __future__ import annotations

import numpy as np
from PIL import Image


def overlay_mask(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.5,
    draw_contour: bool = True,
) -> Image.Image:
    """Blend a binary mask onto the image with the given color and alpha."""
    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image.copy()
    if img.shape[:2] != mask.shape:
        from PIL import Image as PILImage
        mask_im = PILImage.fromarray((mask * 255).astype(np.uint8)).resize(
            (img.shape[1], img.shape[0]), resample=PILImage.NEAREST
        )
        mask = np.array(mask_im) > 127

    mask_b = mask.astype(bool)
    overlay = img.copy()
    overlay[mask_b] = (alpha * np.array(color) + (1 - alpha) * img[mask_b]).astype(np.uint8)

    if draw_contour:
        try:
            import cv2
            edges = cv2.Canny(mask_b.astype(np.uint8) * 255, 100, 200)
            overlay[edges > 0] = color
        except ImportError:
            pass

    return Image.fromarray(overlay)
