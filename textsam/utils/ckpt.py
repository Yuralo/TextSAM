"""Checkpoint save/load helpers.

We save *only* trainable parameters (adapter + decoder) plus optimizer/scheduler
state. The frozen SAM encoder + CLIP text encoder are reloaded from their
upstream pretrained weights at startup, so we don't carry them around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


_COMPILE_PREFIX = "_orig_mod."


def _strip_compile_prefix(name: str) -> str:
    # torch.compile wraps modules and prefixes parameter names with "_orig_mod.".
    # Strip so checkpoints stay portable between compiled and uncompiled runs.
    while _COMPILE_PREFIX in name:
        name = name.replace(_COMPILE_PREFIX, "")
    return name


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        _strip_compile_prefix(n): p.detach().cpu()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


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
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # Re-key to match whatever wrapping the live model has (compiled vs not).
    live_keys = {k for k, _ in model.named_parameters()}
    saved = payload["model"]
    needs_prefix = any(k.startswith(_COMPILE_PREFIX) for k in live_keys)
    if needs_prefix:
        saved = {f"{_COMPILE_PREFIX}{k}" if not k.startswith(_COMPILE_PREFIX) else k: v for k, v in saved.items()}
    else:
        saved = {_strip_compile_prefix(k): v for k, v in saved.items()}
    msg = model.load_state_dict(saved, strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return {"extras": payload.get("extras", {}), "load_msg": str(msg)}
