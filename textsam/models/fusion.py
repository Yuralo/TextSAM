"""Cross-modal adapter: turns CLIP text features into SAM sparse prompt tokens.

Architecture (per layer, repeated `num_layers` times):

  text tokens (B, L, D_t)  -- linear --> D_h
  image tokens (B, HW, D_i) -- linear --> D_h        (HW = 64*64 = 4096)

  for each layer:
      text  <-- cross-attn(query=text, key/value=image)
      text  <-- self-attn + FFN

  finally:
      K learnable query tokens attend over the refined text tokens to produce
      K sparse prompt tokens of dim D_h (=256) that SAM's MaskDecoder consumes
      as `sparse_prompt_embeddings`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, q: Tensor, kv: Tensor, q_mask: Tensor | None = None) -> Tensor:
        # cross-attention: q attends to kv (no mask on kv side -- all image tokens are valid)
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        key_padding_mask = (q_mask == 0) if q_mask is not None else None
        # For nn.MultiheadAttention key_padding_mask applies to *key*; we don't need to mask image keys.
        attn_out, _ = self.cross_attn(q_n, kv_n, kv_n, need_weights=False)
        q = q + attn_out

        # self-attention over text tokens
        q_n = self.norm_sa(q)
        # mask out padded text tokens on the key side
        sa_out, _ = self.self_attn(q_n, q_n, q_n, key_padding_mask=key_padding_mask, need_weights=False)
        q = q + sa_out

        # FFN
        q = q + self.ff(self.norm_ff(q))
        return q


class CrossModalAdapter(nn.Module):
    """CLIP text + SAM image features -> K sparse prompt tokens for SAM mask decoder."""

    def __init__(
        self,
        text_dim: int = 512,
        image_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        num_prompt_tokens: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prompt_tokens = num_prompt_tokens

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Conv2d(image_dim, hidden_dim, kernel_size=1)

        # learned positional embedding over the (64,64) image grid (1024² input).
        # Interpolated on the fly for other grids (e.g. 32×32 at 512² in stage 2).
        self.image_grid = 64
        self.image_pos = nn.Parameter(torch.zeros(1, self.image_grid * self.image_grid, hidden_dim))
        nn.init.trunc_normal_(self.image_pos, std=0.02)

        self.blocks = nn.ModuleList(
            CrossAttnBlock(hidden_dim, num_heads=num_heads) for _ in range(num_layers)
        )

        # K learnable queries that pool the refined text tokens into K SAM prompt tokens.
        self.prompt_queries = nn.Parameter(torch.zeros(1, num_prompt_tokens, hidden_dim))
        nn.init.trunc_normal_(self.prompt_queries, std=0.02)
        self.pool_norm_q = nn.LayerNorm(hidden_dim)
        self.pool_norm_kv = nn.LayerNorm(hidden_dim)
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.pool_ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def _image_pos(self, H: int, W: int) -> Tensor:
        """Learned (64×64) image pos-embed, bicubically resized to the (H,W) grid."""
        g = self.image_grid
        if H == g and W == g:
            return self.image_pos
        pos = self.image_pos.reshape(1, g, g, self.hidden_dim).permute(0, 3, 1, 2)  # (1,D,g,g)
        pos = F.interpolate(pos, size=(H, W), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, H * W, self.hidden_dim)

    def forward(self, image_feat: Tensor, text_tokens: Tensor, text_mask: Tensor) -> Tensor:
        """
        Args:
            image_feat:  (B, C_i, H, W) — SAM image embeddings, typically (B, 256, 64, 64).
            text_tokens: (B, L, D_t)    — CLIP text token states.
            text_mask:   (B, L)         — 1 for real tokens, 0 for padding.

        Returns:
            sparse_prompts: (B, K, 256) — SAM-compatible sparse prompt embeddings.
        """
        B, _, H, W = image_feat.shape

        img = self.image_proj(image_feat)                  # (B, D, H, W)
        img = img.flatten(2).transpose(1, 2)               # (B, HW, D)
        img = img + self._image_pos(H, W)

        txt = self.text_proj(text_tokens)                  # (B, L, D)

        for blk in self.blocks:
            txt = blk(txt, img, q_mask=text_mask)

        # Pool to K tokens using learned queries.
        q = self.prompt_queries.expand(B, -1, -1)          # (B, K, D)
        q_n = self.pool_norm_q(q)
        kv_n = self.pool_norm_kv(txt)
        key_padding_mask = (text_mask == 0)
        pooled, _ = self.pool_attn(q_n, kv_n, kv_n, key_padding_mask=key_padding_mask, need_weights=False)
        prompts = q + pooled
        prompts = prompts + self.pool_ff(prompts)
        return prompts                                     # (B, K, 256)
