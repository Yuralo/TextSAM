# TextSAM — Architecture

TextSAM is an open-vocabulary segmentation model that takes an `(image, word)`
pair and outputs a binary mask of the named object inside the image. It is
built by **stitching three pretrained foundation models together with a small
trainable cross-modal adapter**:

```
                     (frozen)                       (frozen)
   image ───► SAM ViT-B image encoder ──► img_feat (B,256,64,64)
                                                       │
   word/phrase ─► CLIP ViT-B/16 text enc ─► txt_tok (B,L,512)
                                                       │
                                       ┌───────────────┘
                                       ▼
                          Cross-modal Adapter (TRAINABLE, ~6 M)
                          • 2× cross-attention (text ↔ image)
                          • learnable pooling -> K=4 SAM "sparse prompt" tokens (B,4,256)
                                       │
                                       ▼
                          SAM Mask Decoder (TRAINABLE, ~4 M; warm-started)
                          • two-way transformer + hypernet head
                          • outputs: 1 mask (stage 1) or N masks (stage 2, N queries)
```

## Why this design

The original SAM paper (Kirillov et al., 2023 [@kirillov2023sam]) described an
optional **text-prompt** input to its prompt encoder but Meta never released
weights for that path. TextSAM realizes exactly that missing slot — using
CLIP's text encoder (Radford et al., 2021 [@radford2021clip]) as the source of
text embeddings, and a learned cross-modal adapter to fuse text with the SAM
image features before handing off to SAM's pretrained mask decoder.

Compared to other open-vocabulary segmentation approaches:

| Approach | Reference | Trainable params | Stage-1 fit | Stage-2 fit | 3090 fit |
|---|---|---|---|---|---|
| **TextSAM (ours)** | this work | **~11–15 M** | native (1 mask) | re-batch K queries | yes (bs 4 @ 1024²) |
| CLIPSeg | [@luddecke2022clipseg] | ~5–10 M | yes | semantic only | yes |
| Mask2Former + CLIP queries | [@cheng2022mask2former], [@zou2023xdecoder] | ~60–80 M | awkward | native | tight |
| LISA (LLM-driven) | [@lai2024lisa] | LLM + decoder | overkill for "word" | yes | needs >24 GB |
| Grounded-SAM | [@ren2024groundedsam] | 0 (zero-shot pipeline) | yes | yes | yes |
| PerSAM | [@zhang2024persam] | 0 (one-shot prompt tuning) | yes | yes | yes |

The trainable-only variants (Grounded-SAM, PerSAM) avoid training but inherit
the limitations of the upstream detectors. We *learn* the text-prompt path so
the mask quality matches what SAM produces when given a perfect point/box.

## Component details

### Image encoder (`textsam/models/encoders.py`)

`SAMImageEncoder` loads the official `segment_anything` ViT-B model and exposes
its image encoder, prompt encoder (only to fetch the dense positional encoding
`image_pe`), and mask decoder (so we can reuse and finetune its weights).
Image encoder is frozen and kept in `.eval()`.

Output shape: `(B, 256, 64, 64)` for a 1024×1024 input.

### Text encoder (`textsam/models/encoders.py`)

`CLIPTextEncoder` is the HuggingFace `transformers.CLIPTextModel` from
`openai/clip-vit-base-patch16`. Returns the pooled CLS embedding, the per-token
hidden states (`(B, L, 512)`), and the attention mask. Frozen, `.eval()`.

### Cross-modal adapter (`textsam/models/fusion.py`)

```
text_tokens (B, L, 512)  ─linear→  (B, L, 256)
image_feat  (B,256,64,64)─conv1×1→ (B, 4096, 256)  + pos
        ↓
2× CrossAttnBlock:
    text  <-- cross-attn(text, image, image)
    text  <-- self-attn(text)
    text  <-- FFN
        ↓
K=4 learned query tokens attend over refined text tokens
        ↓
sparse_prompts (B, 4, 256) -> SAM mask decoder
```

This is the only fundamentally new component. The cross-attention blocks are
standard pre-norm transformer blocks; the final K=4 prompt tokens mimic the
shape SAM expects from its prompt encoder (one point would be 2 tokens; we
give the decoder more room with 4).

### Mask decoder (`textsam/models/decoder.py`)

A thin wrapper around `segment_anything.modeling.MaskDecoder`. We pass:

- `image_embeddings`: the SAM image features
- `image_pe`: the learned dense positional encoding from SAM's prompt encoder
- `sparse_prompt_embeddings`: the K=4 tokens from our adapter
- `dense_prompt_embeddings`: zeros (no mask hint)
- `multimask_output`: `False` (one mask per forward)

The decoder's output `low_res_masks` is `(B, 1, 256, 256)`. We upsample
bilinearly to the input image size (1024² or 512² depending on stage).

## Training-friendly structure

`TextSAM.parameter_groups(lr_adapter, lr_decoder)` produces two AdamW groups
with different learning rates — adapter gets a higher LR (it's randomly
initialised), decoder gets a lower LR (it's already pretrained). This matches
the practice in CLIPSeg and EVF-SAM.

## Loss

Combined Dice + Focal + IoU-prediction MSE
(`textsam/train/losses.py:combined_seg_loss`):

- Dice loss (Milletari et al., 2016 [@milletari2016vnet]) on sigmoid probabilities.
- Sigmoid focal loss (Lin et al., 2017 [@lin2017focalloss]) with α=0.25, γ=2.
- IoU prediction MSE matches SAM's auxiliary head.

Weights: 0.5 Dice + 0.5 Focal + 0.05 IoU. Stage 2's instance variant adds an
optional `HungarianMatcher` (`textsam/train/losses.py:HungarianMatcher`) to
pair predicted query masks with ground-truth instances by minimum
Dice+Focal cost.

## Trainable-parameter budget (precise)

Approximate counts at the configured dimensions (`hidden_dim=256`,
`num_layers=2`, `num_heads=8`, `num_prompt_tokens=4`):

| Module | Params |
|---|---|
| Text projection (Linear 512→256) | 0.13 M |
| Image projection (Conv 256→256, 1×1) | 0.07 M |
| Image positional embedding (4096×256) | 1.05 M |
| 2× CrossAttnBlock | ~3.5 M |
| K=4 prompt query pool + LayerNorms + FFN | 0.7 M |
| SAM mask decoder | ~4.0 M |
| **Total** | **~9.5 M** |

This is comfortably under the 25–30 M loose upper-bound and leaves headroom
for adding adapter layers or expanding `num_prompt_tokens` if quality demands.

## File map

| Concern | File |
|---|---|
| Frozen encoders | `textsam/models/encoders.py` |
| Cross-modal fusion | `textsam/models/fusion.py` |
| Mask decoder wrapper | `textsam/models/decoder.py` |
| Full model | `textsam/models/textsam.py` |
| Losses | `textsam/train/losses.py` |
| Datasets | `textsam/data/{phrasecut,ade20k,lvis,merged}.py` |
| Augmentation + SAM preproc | `textsam/data/transforms.py` |
| Trainer | `textsam/train/trainer.py` |
| Inference CLI | `textsam/inference/predict.py` |

## References

Citations are kept in `docs/citations.bib`. See `[@key]` markers above for
mapping to the BibTeX entries.
