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


def _resize_sam_for_input(encoder, prompt_encoder, new_img_size: int, patch: int = 16):
    """Adapt a 1024²-pretrained SAM ViT to a different square input size.

    SAM ViT-B's absolute pos_embed is a fixed 64×64 grid and its global-attention
    blocks carry rel_pos tables sized to that grid. For a smaller input we
    interpolate both, plus the prompt encoder's dense-PE grid, so the whole stack
    is internally consistent at the new resolution. Window-attention blocks use a
    fixed 14×14 window and need no change.
    """
    old_img_size = encoder.img_size
    if new_img_size == old_img_size:
        return
    new_grid = new_img_size // patch

    # 1) absolute pos_embed: (1, gh, gw, C) -> (1, new_grid, new_grid, C)
    pe = encoder.pos_embed.data.permute(0, 3, 1, 2)          # (1, C, gh, gw)
    pe = F.interpolate(pe, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    encoder.pos_embed = nn.Parameter(pe.permute(0, 2, 3, 1).contiguous())

    # 2) global-attention rel_pos tables (window_size == 0 blocks), length 2*grid-1
    new_len = 2 * new_grid - 1
    for blk in encoder.blocks:
        if getattr(blk, "window_size", 0) == 0 and getattr(blk.attn, "use_rel_pos", False):
            for name in ("rel_pos_h", "rel_pos_w"):
                rp = getattr(blk.attn, name).data                # (2*old_grid-1, head_dim)
                rp = rp.permute(1, 0).unsqueeze(0)               # (1, head_dim, L_old)
                rp = F.interpolate(rp, size=new_len, mode="linear", align_corners=False)
                rp = rp.squeeze(0).permute(1, 0).contiguous()    # (L_new, head_dim)
                setattr(blk.attn, name, nn.Parameter(rp))

    encoder.img_size = new_img_size
    # 3) prompt encoder dense-PE grid + nominal input size
    prompt_encoder.image_embedding_size = (new_grid, new_grid)
    prompt_encoder.input_image_size = (new_img_size, new_img_size)


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

    def __init__(self, model_type: str = "sam_vit_b", checkpoint: str | None = None, image_size: int = 1024):
        super().__init__()
        _patch_sam_attention_with_sdpa()
        sam = _build_sam(model_type, checkpoint)
        self.encoder = sam.image_encoder
        # Held so callers can fetch the dense positional encoding and re-init the
        # mask decoder. We do NOT freeze these — the caller decides what to freeze.
        self._sam_prompt_encoder = sam.prompt_encoder
        self._sam_mask_decoder = sam.mask_decoder
        # Stage 2 runs at 512²; interpolate SAM's 1024²-pretrained pos embeddings.
        if image_size != self.encoder.img_size:
            _resize_sam_for_input(self.encoder, self._sam_prompt_encoder, image_size)
        self.image_size = self.encoder.img_size

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

    def __init__(self, name: str = "openai/clip-vit-base-patch16", max_length: int = 32, cache: bool = False):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizerFast

        self.tokenizer = CLIPTokenizerFast.from_pretrained(name)
        self.model = CLIPTextModel.from_pretrained(name)
        self.max_length = max_length
        self.hidden_dim = self.model.config.hidden_size  # 512 for ViT-B/16
        # Per-string embedding cache. Only valid when the encoder is frozen and the
        # vocabulary is closed (stage 2: a fixed set of class names that repeats
        # every step). Stores (tokens[L,D], mask[L], pooled[D]) on the model device.
        self._cache_enabled = cache
        self._cache: dict[str, tuple[Tensor, Tensor, Tensor]] = {}

    def enable_cache(self, enabled: bool = True):
        self._cache_enabled = enabled
        if not enabled:
            self._cache.clear()

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

    @torch.no_grad()
    def _encode_uncached(self, texts: List[str], device):
        inputs = self.tokenize(texts, device=device)
        out = self.model(**inputs)
        return out.pooler_output, out.last_hidden_state, inputs["attention_mask"]

    def _forward_cached(self, texts: List[str]) -> tuple[Tensor, Tensor, Tensor]:
        device = next(self.model.parameters()).device
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            pooled, tokens, mask = self._encode_uncached(missing, device)
            for i, t in enumerate(missing):
                self._cache[t] = (tokens[i], mask[i], pooled[i])
        tokens = torch.stack([self._cache[t][0] for t in texts])
        mask = torch.stack([self._cache[t][1] for t in texts])
        pooled = torch.stack([self._cache[t][2] for t in texts])
        return pooled, tokens, mask

    def forward(self, texts: List[str] | dict) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (pooled[B,D], tokens[B,L,D], mask[B,L])."""
        if self._cache_enabled and isinstance(texts, list):
            return self._forward_cached(texts)
        if isinstance(texts, list):
            device = next(self.model.parameters()).device
            inputs = self.tokenize(texts, device=device)
        else:
            inputs = texts
        out = self.model(**inputs)
        return out.pooler_output, out.last_hidden_state, inputs["attention_mask"]

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
