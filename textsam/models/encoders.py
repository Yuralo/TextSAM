"""Frozen image (SAM) and text (CLIP) encoders.

Both backbones are loaded with pretrained weights and set to eval mode with
gradients disabled. They are *not* trained — only the cross-modal adapter
and the SAM mask decoder are updated.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -------------------- SAM image encoder --------------------

_SDPA_PATCHED = False


def _patch_sam_attention_with_sdpa() -> bool:
    """Swap segment_anything's manual softmax attention for SDPA.

    SDPA dispatches to Flash Attention (or memory-efficient attention when a
    bias is present) and is ~1.3-1.8x faster on Ampere with the same numerics.
    Idempotent — safe to call multiple times.
    """
    global _SDPA_PATCHED
    if _SDPA_PATCHED:
        return True
    try:
        from segment_anything.modeling.image_encoder import (
            Attention,
            add_decomposed_rel_pos,
        )
    except ImportError:
        return False

    def sdpa_forward(self, x: Tensor) -> Tensor:
        B, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn_bias = None
        if self.use_rel_pos:
            attn_bias = torch.zeros(
                B * self.num_heads, H * W, H * W, dtype=q.dtype, device=q.device
            )
            attn_bias = add_decomposed_rel_pos(
                attn_bias, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
        # SDPA does its own (q @ k.T) * scale internally — drop the manual scale.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = (
            out.view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        return self.proj(out)

    Attention.forward = sdpa_forward
    _SDPA_PATCHED = True
    return True


def _build_sam(model_type: str, checkpoint: str | None):
    """Build a SAM model via the official `segment_anything` package."""
    try:
        from segment_anything import sam_model_registry
    except ImportError as e:
        raise ImportError(
            "segment_anything not installed. Run: "
            "pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from e

    builder = sam_model_registry[model_type.replace("sam_", "")]
    sam = builder(checkpoint=checkpoint if checkpoint and Path(checkpoint).exists() else None)
    return sam


class SAMImageEncoder(nn.Module):
    """Wraps SAM's ViT image encoder. Outputs (B, 256, 64, 64) features.

    The full SAM model also provides the prompt encoder and mask decoder; we
    keep references to them so the TextSAM model can pull `image_pe` and reuse
    the pretrained MaskDecoder weights.
    """

    def __init__(self, model_type: str = "sam_vit_b", checkpoint: str | None = None):
        super().__init__()
        _patch_sam_attention_with_sdpa()
        sam = _build_sam(model_type, checkpoint)
        self.encoder = sam.image_encoder
        # Held so callers can fetch the dense positional encoding (1,256,64,64)
        # and re-init the mask decoder. We do NOT freeze these — the caller
        # decides what to freeze and what to train.
        self._sam_prompt_encoder = sam.prompt_encoder
        self._sam_mask_decoder = sam.mask_decoder
        self.image_size = sam.image_encoder.img_size  # 1024

    def forward(self, images: Tensor) -> Tensor:
        return self.encoder(images)

    def image_pe(self) -> Tensor:
        """SAM's learned dense positional encoding: (1, 256, 64, 64)."""
        return self._sam_prompt_encoder.get_dense_pe()

    @property
    def pretrained_mask_decoder(self) -> nn.Module:
        return self._sam_mask_decoder

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()


# -------------------- CLIP text encoder --------------------

class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder (HuggingFace `transformers`).

    Returns both the pooled (CLS-equivalent) embedding and the token-wise hidden
    states so the cross-modal adapter can attend over tokens.
    """

    def __init__(self, name: str = "openai/clip-vit-base-patch16", max_length: int = 32):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizerFast

        self.tokenizer = CLIPTokenizerFast.from_pretrained(name)
        self.model = CLIPTextModel.from_pretrained(name)
        self.max_length = max_length
        self.hidden_dim = self.model.config.hidden_size  # 512 for ViT-B/16

    @torch.no_grad()
    def tokenize(self, texts: List[str], device: torch.device | str = "cpu"):
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(self, texts: List[str] | dict) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (pooled[B,D], tokens[B,L,D], mask[B,L])."""
        if isinstance(texts, list):
            device = next(self.model.parameters()).device
            inputs = self.tokenize(texts, device=device)
        else:
            inputs = texts
        out = self.model(**inputs)
        pooled = out.pooler_output            # (B, D)
        tokens = out.last_hidden_state        # (B, L, D)
        mask = inputs["attention_mask"]       # (B, L)
        return pooled, tokens, mask

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
