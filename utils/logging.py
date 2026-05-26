"""Tiny TensorBoard wrapper that's safe to use even if tensorboard isn't installed."""

from __future__ import annotations

from pathlib import Path


class TBLogger:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except Exception:
            self.writer = None

    def log_scalar(self, name: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def log_scalars(self, prefix: str, mapping: dict[str, float], step: int):
        for k, v in mapping.items():
            self.log_scalar(f"{prefix}/{k}", v, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
