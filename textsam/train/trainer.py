"""Training loop for TextSAM (both stages).

Handles:
  - bf16 mixed-precision via torch.amp
  - gradient accumulation
  - cosine LR schedule with linear warmup
  - checkpoint save/load (best by val mIoU, plus last-K)
  - tqdm + tensorboard logging
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.ckpt import load_checkpoint, save_checkpoint
from ..utils.logging import TBLogger
from ..utils.metrics import miou
from .losses import combined_seg_loss


def cosine_lr_with_warmup(step: int, total: int, warmup: int, min_ratio: float = 0.01) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * p))
    return max(min_ratio, cos)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: dict,
        device: torch.device | str = "cuda",
        step_fn: Callable | None = None,   # custom step (used by stage 2 multi-query)
    ):
        # Free wins: TF32 for fp32 matmuls + cuDNN autotuner for fixed shapes.
        # Both are safe for SAM/CLIP at fixed input resolution.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device(device)
        self.step_fn = step_fn or self._stage1_step

        tcfg = cfg["train"]
        ocfg = tcfg["optimizer"]
        scfg = tcfg["scheduler"]

        self.optim = torch.optim.AdamW(
            model.parameter_groups(lr_adapter=ocfg["lr_adapter"], lr_decoder=ocfg["lr_decoder"]),
            weight_decay=ocfg["weight_decay"],
        )
        self.epochs = tcfg["epochs"]
        self.grad_accum = tcfg.get("grad_accum_steps", 1)
        self.amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[tcfg.get("amp_dtype", "bfloat16")]
        # We don't need GradScaler for bf16; only fp16 needs scaling.
        self.use_scaler = self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda") if self.use_scaler else None

        self.total_iters = self.epochs * len(train_loader)
        self.warmup = scfg.get("warmup_iters", 0)
        self.min_lr_ratio = scfg.get("min_lr_ratio", 0.01)
        self.base_lrs = [g["lr"] for g in self.optim.param_groups]

        lcfg = cfg["logging"]
        self.log_dir = Path(lcfg["log_dir"])
        self.ckpt_dir = Path(lcfg["ckpt_dir"])
        self.save_every = lcfg.get("save_every_n_epochs", 1)
        self.save_every_steps = lcfg.get("save_every_n_steps", 0)  # 0 disables intra-epoch saves
        self.keep_last = lcfg.get("keep_last", 3)
        self.recent_ckpts: deque[Path] = deque()
        self.logger = TBLogger(self.log_dir)
        self.loss_cfg = cfg["loss"]
        self.best_val = -float("inf")
        self.global_step = 0
        self.start_epoch = 1

    # ---------- per-step ----------

    def _set_lr(self, step: int):
        scale = cosine_lr_with_warmup(step, self.total_iters, self.warmup, self.min_lr_ratio)
        for g, base in zip(self.optim.param_groups, self.base_lrs):
            g["lr"] = base * scale

    def _stage1_step(self, batch: dict) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        texts = batch["text"]
        with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
            mask_logits, iou_pred = self.model(images, texts)
            loss, parts = combined_seg_loss(
                mask_logits, masks, iou_pred,
                dice_weight=self.loss_cfg["dice_weight"],
                focal_weight=self.loss_cfg["focal_weight"],
                focal_alpha=self.loss_cfg["focal_alpha"],
                focal_gamma=self.loss_cfg["focal_gamma"],
                iou_weight=self.loss_cfg["iou_weight"],
            )
        return loss, parts, mask_logits, masks

    # ---------- loops ----------

    def fit(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_one_epoch(epoch)
            if self.val_loader is not None and epoch % self.cfg["eval"].get("every_n_epochs", 1) == 0:
                self._validate(epoch)
            if epoch % self.save_every == 0:
                self._save(epoch, is_best=False)
            # Always refresh `last.pt` at epoch boundary so a crashed run
            # never loses more than one epoch + save_every_n_steps.
            self._save_latest(epoch)
        self.logger.close()

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        # keep frozen submodules in eval to disable dropout/BN updates
        if hasattr(self.model, "image_encoder"):
            self.model.image_encoder.eval()
        if hasattr(self.model, "text_encoder"):
            self.model.text_encoder.model.eval()

        running = {"total": 0.0, "dice": 0.0, "focal": 0.0, "iou": 0.0}
        pbar = tqdm(self.train_loader, desc=f"train epoch {epoch}/{self.epochs}", leave=False)
        self.optim.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            self._set_lr(self.global_step)
            loss, parts, _, _ = self.step_fn(batch)
            loss = loss / self.grad_accum
            if self.use_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if (i + 1) % self.grad_accum == 0:
                if self.use_scaler:
                    self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(list(self.model.trainable_parameters()), max_norm=1.0)
                if self.use_scaler:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()
                self.optim.zero_grad(set_to_none=True)

            for k in running:
                running[k] = 0.95 * running[k] + 0.05 * parts.get(k, 0.0)
            self.global_step += 1
            pbar.set_postfix(loss=f"{running['total']:.4f}", dice=f"{running['dice']:.3f}", lr=f"{self.optim.param_groups[0]['lr']:.2e}")
            self.logger.log_scalars("train", parts, self.global_step)

            if self.save_every_steps and self.global_step % self.save_every_steps == 0:
                self._save_latest(epoch)
                print(f"[ckpt] last.pt updated at step {self.global_step} (epoch {epoch})")

    @torch.no_grad()
    def _validate(self, epoch: int):
        self.model.eval()
        ious = []
        for batch in tqdm(self.val_loader, desc=f"val epoch {epoch}", leave=False):
            _, _, logits, masks = self.step_fn(batch)
            ious.append(miou(logits.float(), masks))
        m = float(sum(ious) / max(1, len(ious)))
        print(f"[val epoch {epoch}] mIoU = {m:.4f}")
        self.logger.log_scalar("val/miou", m, epoch)
        if m > self.best_val:
            self.best_val = m
            self._save(epoch, is_best=True)

    # ---------- ckpt ----------

    def _extras(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val": self.best_val,
        }

    def _save(self, epoch: int, is_best: bool):
        name = "best.pt" if is_best else f"epoch_{epoch:03d}.pt"
        path = self.ckpt_dir / name
        save_checkpoint(path, self.model, self.optim, scaler=self.scaler, extras=self._extras(epoch))
        if not is_best:
            self.recent_ckpts.append(path)
            while len(self.recent_ckpts) > self.keep_last:
                old = self.recent_ckpts.popleft()
                if old.exists():
                    old.unlink()
        print(f"[ckpt] saved {path}")

    def _save_latest(self, epoch: int):
        """Atomic-ish update of `last.pt` for resume — overwrites every save."""
        path = self.ckpt_dir / "last.pt"
        tmp = path.with_suffix(".pt.tmp")
        save_checkpoint(tmp, self.model, self.optim, scaler=self.scaler, extras=self._extras(epoch))
        tmp.replace(path)

    def load(self, path: str | Path, resume: bool = False):
        """Load weights. If `resume`, also restore optimizer + step + epoch."""
        info = load_checkpoint(
            path,
            self.model,
            optimizer=self.optim if resume else None,
            scaler=self.scaler if resume else None,
            strict=False,
        )
        if resume:
            ex = info.get("extras", {})
            self.global_step = int(ex.get("global_step", 0))
            self.best_val = float(ex.get("best_val", -float("inf")))
            self.start_epoch = int(ex.get("epoch", 0)) + 1
            print(
                f"[ckpt] resumed {path}: epoch {self.start_epoch-1} done, "
                f"step {self.global_step}, best_val {self.best_val:.4f}"
            )
        else:
            print(f"[ckpt] loaded {path}: {info['load_msg']}")
