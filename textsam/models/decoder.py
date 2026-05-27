"""Thin wrapper around SAM's MaskDecoder.

We reuse the pretrained `MaskDecoder` weights from the SAM checkpoint (loaded
inside SAMImageEncoder._sam_mask_decoder). The decoder is fed:

  - image_embeddings: (B, 256, 64, 64) from the frozen SAM image encoder
  - image_pe:         (1, 256, 64, 64) from SAM's PromptEncoder
  - sparse_prompts:   (B, K, 256)      from our CrossModalAdapter
  - dense_prompts:    (B, 256, 64, 64) zeros (we have no mask hint)

and returns:

  - low_res_masks: (B, 1, 256, 256)
  - iou_pred:      (B, 1)

We upsample low_res_masks bilinearly to the input image size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TextSAMDecoder(nn.Module):
    def __init__(
        self,
        sam_mask_decoder: nn.Module,
        no_mask_embed: nn.Embedding | None = None,
        multimask_output: bool = False,
    ):
        super().__init__()
        self.decoder = sam_mask_decoder
        self.multimask_output = multimask_output
        # Reuse SAM's learned "no mask hint" embedding so the decoder sees the
        # exact dense-prompt distribution it was pretrained against. Falls back
        # to zeros if the prompt encoder isn't available (e.g., in self-test).
        if no_mask_embed is not None:
            self.no_mask_embed = no_mask_embed
        else:
            self.no_mask_embed = nn.Embedding(1, 256)
            nn.init.zeros_(self.no_mask_embed.weight)

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompts: Tensor,
        output_size: int = 1024,
    ) -> tuple[Tensor, Tensor]:
        B = image_embeddings.shape[0]
        H, W = image_embeddings.shape[-2:]
        dense_per_image = (
            self.no_mask_embed.weight.reshape(1, -1, 1, 1)
            .expand(1, -1, H, W)
            .to(image_embeddings.dtype)
        )

        # SAM's MaskDecoder is implemented for single-image / multi-prompt: it
        # does `repeat_interleave(image_embeddings, tokens.shape[0])` internally,
        # which produces B*B instead of B when image_embeddings is batched.
        # Call it per-batch-item and stack to keep gradients clean.
        masks_low_list, iou_list = [], []
        for i in range(B):
            m_i, iou_i = self.decoder(
                image_embeddings=image_embeddings[i : i + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompts[i : i + 1],
                dense_prompt_embeddings=dense_per_image,
                multimask_output=self.multimask_output,
            )
            masks_low_list.append(m_i)
            iou_list.append(iou_i)
        low_res_masks = torch.cat(masks_low_list, dim=0)
        iou_pred = torch.cat(iou_list, dim=0)

        # low_res_masks: (B, num_masks, 256, 256). With multimask_output=False, num_masks=1.
        masks = F.interpolate(
            low_res_masks,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )
        return masks, iou_pred
