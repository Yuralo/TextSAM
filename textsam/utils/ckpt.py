"""Checkpoint save/load helpers.

We save *only* trainable parameters (adapter + decoder) plus optimizer/scheduler
state. The frozen SAM encoder + CLIP text encoder are reloaded from their
upstream pretrained weights at startup, so we don't carry them around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    extras: dict[str, Any] | None = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extras": extras or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    strict: bool = False,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    msg = model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return {"extras": payload.get("extras", {}), "load_msg": str(msg)}
