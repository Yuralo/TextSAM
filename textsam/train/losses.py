"""Segmentation losses.

- `dice_loss`: soft Dice loss on sigmoid probabilities. Milletari et al., V-Net (3DV 2016).
- `sigmoid_focal_loss`: dense focal loss. Lin et al., ICCV 2017.
- `combined_seg_loss`: weighted sum used in stages 1 and 2.
- `HungarianMatcher`: optional matcher for the LVIS instance-variant of stage 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def dice_loss(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    """Soft Dice over flattened (B, *) tensors. Returns scalar."""
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1).float()
    inter = (probs * targets).sum(-1)
    denom = probs.sum(-1) + targets.sum(-1)
    dice = (2 * inter + eps) / (denom + eps)
    return (1 - dice).mean()


def sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Per-pixel focal loss, averaged. Mirrors torchvision.ops.sigmoid_focal_loss with mean reduction."""
    p = torch.sigmoid(logits)
    targets = targets.float()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def iou_pred_loss(pred_iou: Tensor, logits: Tensor, targets: Tensor) -> Tensor:
    """MSE between SAM's predicted IoU and the actual IoU of the produced mask."""
    with torch.no_grad():
        probs = (torch.sigmoid(logits) > 0.5).float()
        inter = (probs * targets).flatten(1).sum(-1)
        union = ((probs + targets) > 0).float().flatten(1).sum(-1).clamp(min=1.0)
        actual_iou = inter / union  # (B,)
    return F.mse_loss(pred_iou.squeeze(-1), actual_iou)


def combined_seg_loss(
    logits: Tensor,
    targets: Tensor,
    pred_iou: Tensor | None,
    dice_weight: float = 0.5,
    focal_weight: float = 0.5,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    iou_weight: float = 0.05,
) -> tuple[Tensor, dict]:
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    d = dice_loss(logits, targets)
    f = sigmoid_focal_loss(logits, targets, alpha=focal_alpha, gamma=focal_gamma)
    total = dice_weight * d + focal_weight * f
    parts = {"dice": float(d.detach()), "focal": float(f.detach())}
    if pred_iou is not None and iou_weight > 0:
        i = iou_pred_loss(pred_iou, logits, targets)
        total = total + iou_weight * i
        parts["iou"] = float(i.detach())
    parts["total"] = float(total.detach())
    return total, parts


# -------------------- Hungarian matcher (stage 2 instance variant) --------------------

class HungarianMatcher(nn.Module):
    """Match K predicted query masks to M ground-truth instances by minimum
    cost = w_dice * dice + w_focal * focal. Returns indices for each batch.
    """

    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    @torch.no_grad()
    def forward(self, pred_logits: Tensor, gt_masks: Tensor):
        """
        pred_logits: (B, K, H, W) raw logits.
        gt_masks:    list of B tensors, each (M_b, H, W).
        Returns: list of (idx_pred, idx_gt) tuples of length B.
        """
        from scipy.optimize import linear_sum_assignment

        B, K, H, W = pred_logits.shape
        out = []
        for b in range(B):
            preds = pred_logits[b].sigmoid().flatten(1)        # (K, HW)
            gts = gt_masks[b].float().flatten(1)               # (M, HW)
            M = gts.shape[0]
            if M == 0:
                out.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
                continue
            # dice cost (K, M)
            inter = preds @ gts.t()
            denom = preds.sum(-1, keepdim=True) + gts.sum(-1, keepdim=True).t()
            dice_cost = 1 - (2 * inter + 1e-6) / (denom + 1e-6)
            # focal-like cost using mean BCE
            # (K,M) = mean_{HW} BCE(preds[k], gts[m]); approximate via -log per pixel
            p = preds.unsqueeze(1)                              # (K,1,HW)
            t = gts.unsqueeze(0)                                # (1,M,HW)
            focal_cost = -(t * torch.log(p.clamp(min=1e-6)) + (1 - t) * torch.log((1 - p).clamp(min=1e-6))).mean(-1)
            cost = self.dice_weight * dice_cost + self.focal_weight * focal_cost
            row, col = linear_sum_assignment(cost.cpu().numpy())
            out.append((torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)))
        return out
