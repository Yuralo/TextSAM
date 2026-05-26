"""Segmentation metrics: mask IoU and mean IoU.

mIoU here = mean over the dataset of per-sample IoU between the predicted
binary mask (sigmoid > 0.5) and ground-truth binary mask. This is the
standard metric reported on PhraseCut and on multi-query ADE20K eval.
"""

from __future__ import annotations

import torch
from torch import Tensor


def mask_iou(pred_logits: Tensor, target: Tensor, threshold: float = 0.0) -> Tensor:
    """Per-sample IoU. Inputs may be (B, H, W), (B, 1, H, W) or (B, K, 1, H, W) etc.

    Returns a flat tensor of IoUs.
    """
    if pred_logits.ndim == 4 and pred_logits.shape[1] == 1:
        pred_logits = pred_logits.squeeze(1)
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    pred = (pred_logits > threshold).float().flatten(1)
    tgt = (target > 0.5).float().flatten(1)
    inter = (pred * tgt).sum(-1)
    union = ((pred + tgt) > 0).float().sum(-1).clamp(min=1.0)
    return inter / union


def miou(pred_logits: Tensor, target: Tensor, threshold: float = 0.0) -> float:
    return float(mask_iou(pred_logits, target, threshold).mean().detach().cpu())
