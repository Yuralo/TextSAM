"""TextSAM — frozen SAM image encoder + frozen CLIP text encoder
+ trainable CrossModalAdapter + trainable SAM MaskDecoder (warm-started).

The forward signature is `(images, texts) -> mask_logits`. For multi-query
inference (one image with several class names), batch the same image with
each query in the text list, or use `predict_multi`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn
import yaml
from torch import Tensor

from .decoder import TextSAMDecoder
from .encoders import CLIPTextEncoder, SAMImageEncoder
from .fusion import CrossModalAdapter


class TextSAM(nn.Module):
    def __init__(
        self,
        image_encoder: SAMImageEncoder,
        text_encoder: CLIPTextEncoder,
        adapter: CrossModalAdapter,
        decoder: TextSAMDecoder,
        image_size: int = 1024,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.adapter = adapter
        self.decoder = decoder
        self.image_size = image_size

    # ---------- construction ----------

    @classmethod
    def from_config(cls, cfg: dict | str | Path) -> "TextSAM":
        if not isinstance(cfg, dict):
            cfg = yaml.safe_load(Path(cfg).read_text())

        ie_cfg = cfg["image_encoder"]
        te_cfg = cfg["text_encoder"]
        ad_cfg = cfg["adapter"]
        dc_cfg = cfg.get("decoder", {})

        image_enc = SAMImageEncoder(model_type=ie_cfg["name"], checkpoint=ie_cfg.get("checkpoint"))
        if ie_cfg.get("freeze", True):
            image_enc.freeze()

        text_enc = CLIPTextEncoder(name=te_cfg["name"], max_length=te_cfg.get("max_length", 32))
        if te_cfg.get("freeze", True):
            text_enc.freeze()

        adapter = CrossModalAdapter(
            text_dim=ad_cfg.get("text_dim", text_enc.hidden_dim),
            image_dim=ad_cfg.get("image_dim", 256),
            hidden_dim=ad_cfg.get("hidden_dim", 256),
            num_layers=ad_cfg.get("num_layers", 2),
            num_heads=ad_cfg.get("num_heads", 8),
            num_prompt_tokens=ad_cfg.get("num_prompt_tokens", 4),
        )

        # Reuse SAM's learned "no-mask" dense-prompt embedding so the decoder
        # receives exactly the conditioning distribution it was pretrained against.
        no_mask_embed = getattr(image_enc._sam_prompt_encoder, "no_mask_embed", None)
        decoder = TextSAMDecoder(
            sam_mask_decoder=image_enc.pretrained_mask_decoder,
            no_mask_embed=no_mask_embed,
            multimask_output=dc_cfg.get("multimask_output", False),
        )

        return cls(image_enc, text_enc, adapter, decoder, image_size=ie_cfg.get("image_size", 1024))

    # ---------- forward ----------

    def encode_image(self, images: Tensor) -> Tensor:
        """images: (B, 3, H, W) in [0,1] or pre-normalized. Returns (B, 256, 64, 64)."""
        # SAM expects images normalized with its own mean/std and resized to 1024x1024;
        # we assume the dataloader has already done that via `transforms.SAMPreprocess`.
        if self.image_encoder.encoder.training:
            return self.image_encoder(images)
        with torch.no_grad():
            return self.image_encoder(images)

    def encode_text(self, texts: List[str]):
        if any(p.requires_grad for p in self.text_encoder.model.parameters()):
            pooled, tokens, mask = self.text_encoder(texts)
        else:
            with torch.no_grad():
                pooled, tokens, mask = self.text_encoder(texts)
        return pooled, tokens, mask

    def forward(self, images: Tensor, texts: List[str]) -> tuple[Tensor, Tensor]:
        """Returns (mask_logits[B,1,H,W], iou_pred[B,1])."""
        image_feat = self.encode_image(images)
        _, text_tokens, text_mask = self.encode_text(texts)
        sparse_prompts = self.adapter(image_feat, text_tokens, text_mask)
        image_pe = self.image_encoder.image_pe()
        masks, iou = self.decoder(
            image_embeddings=image_feat,
            image_pe=image_pe,
            sparse_prompts=sparse_prompts,
            output_size=self.image_size,
        )
        return masks, iou

    def forward_multi_query(self, images: Tensor, texts_per_image: List[List[str]]) -> tuple[Tensor, Tensor]:
        """Stage-2 forward: each image has a list of K text queries.

        We tile the image embedding K times (cheap — already computed once per image)
        and run the adapter+decoder K times per image, batched as B*K samples.

        Args:
            images: (B, 3, H, W)
            texts_per_image: list of B lists, each of length K.

        Returns:
            masks: (B, K, 1, H, W)
            iou:   (B, K)
        """
        B = images.shape[0]
        K = len(texts_per_image[0])
        assert all(len(t) == K for t in texts_per_image), "all images must share K"

        image_feat = self.encode_image(images)                          # (B, 256, 64, 64)
        image_feat_rep = image_feat.repeat_interleave(K, dim=0)         # (B*K, 256, 64, 64)

        flat_texts: List[str] = [t for ts in texts_per_image for t in ts]
        _, text_tokens, text_mask = self.encode_text(flat_texts)

        sparse_prompts = self.adapter(image_feat_rep, text_tokens, text_mask)
        image_pe = self.image_encoder.image_pe()
        masks, iou = self.decoder(
            image_embeddings=image_feat_rep,
            image_pe=image_pe,
            sparse_prompts=sparse_prompts,
            output_size=self.image_size,
        )
        # reshape (B*K, 1, H, W) -> (B, K, 1, H, W)
        masks = masks.view(B, K, *masks.shape[1:])
        iou = iou.view(B, K)
        return masks, iou

    # ---------- training helpers ----------

    def parameter_groups(self, lr_adapter: float, lr_decoder: float):
        return [
            {"params": self.adapter.parameters(), "lr": lr_adapter, "name": "adapter"},
            {"params": self.decoder.parameters(), "lr": lr_decoder, "name": "decoder"},
        ]

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield p

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())


# -------------------- self-test --------------------

def _self_test():
    """Builds the model with random encoders (no pretrained download) and
    runs a single forward to assert shapes and finite outputs."""
    import os
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

    cfg = {
        "image_encoder": {"name": "sam_vit_b", "checkpoint": None, "freeze": True, "image_size": 1024},
        "text_encoder":  {"name": "openai/clip-vit-base-patch16", "freeze": True, "max_length": 16},
        "adapter": {"text_dim": 512, "image_dim": 256, "hidden_dim": 256, "num_layers": 2, "num_heads": 8, "num_prompt_tokens": 4},
        "decoder": {"multimask_output": False},
    }
    model = TextSAM.from_config(cfg).eval()
    print(f"Trainable params: {model.count_trainable_params()/1e6:.2f} M")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    images = torch.zeros(1, 3, 1024, 1024, device=device)
    masks, iou = model(images, ["a cat"])
    assert masks.shape == (1, 1, 1024, 1024), masks.shape
    assert torch.isfinite(masks).all(), "non-finite mask values"
    print(f"OK: mask {tuple(masks.shape)}  iou {tuple(iou.shape)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        _self_test()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
