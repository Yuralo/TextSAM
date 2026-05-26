# Training

TextSAM is trained in two stages on a single RTX 3090 (24 GB).

## Stage 1 — single-mask, PhraseCut

`configs/stage1_phrasecut.yaml`. The model gets one image + one phrase and
predicts one binary mask. This is exactly the shape SAM's mask decoder was
designed for, so the decoder warm-starts well from its pretrained weights and
the new cross-modal adapter learns "translate phrase → SAM sparse prompt".

```
bash scripts/train_stage1.sh
# or
python -m textsam.train.stage1 --config configs/stage1_phrasecut.yaml
```

### Stage 1 hyperparameters

| | |
|---|---|
| Resolution | 1024² (SAM-native) |
| Batch size | 4 |
| Grad accumulation | 2 → effective batch 8 |
| Optimizer | AdamW(weight_decay=0.05) |
| LR (adapter) | 1e-4 |
| LR (decoder) | 1e-5 |
| Schedule | linear warmup 1 000 iters → cosine to 1 % of base |
| Mixed precision | bf16 (Ampere native — no GradScaler needed) |
| Epochs | 40 |
| Augmentation | flip(0.5) + color-jitter(0.2) + RRC(scale 0.7–1.0) |
| Loss | 0.5 · Dice + 0.5 · Focal(α=0.25, γ=2) + 0.05 · IoU-MSE |
| Wall-clock (3090) | ~2–3 days |
| Validation metric | mIoU on PhraseCut val |

## Stage 2 — multi-query, ADE20K + LVIS

`configs/stage2_ade_lvis.yaml`. We warm-start from the best stage-1 checkpoint
and finetune with K=8 text queries per image (mix of present and absent
classes), each producing one binary mask via `TextSAM.forward_multi_query`.

```
bash scripts/train_stage2.sh
# or
python -m textsam.train.stage2 --config configs/stage2_ade_lvis.yaml
```

### Stage 2 hyperparameters

| | |
|---|---|
| Resolution | 512² (smaller to fit K=8 queries/image) |
| Queries per image | 8 (25 % drawn from class names absent in the image — teaches "refuse" behaviour) |
| Batch size | 4 → 32 effective seg samples per step |
| Grad accumulation | 2 |
| Grad checkpointing | on the mask decoder (decoder runs K times per image) |
| Optimizer | AdamW(weight_decay=0.05) |
| LR (adapter) | 5e-5 |
| LR (decoder) | 5e-6 (lower — already specialised) |
| Schedule | linear warmup 500 → cosine to 1 % |
| Mixed precision | bf16 |
| Epochs | 30 |
| Loss | same as stage 1, per-query, averaged |
| Sampler | `WeightedRandomSampler` over the merged dataset so ADE20K isn't drowned by LVIS |
| Wall-clock (3090) | ~3–5 days |
| Validation metric | mIoU on ADE20K val (semantic) |

For the **instance variant** on LVIS, set `loss.hungarian: true` in the config
to enable the Hungarian matcher in `textsam/train/losses.py:HungarianMatcher`
which assigns each predicted query to a ground-truth instance by minimum
Dice+Focal cost. This adds mask-AP@.5 / .75 metrics during eval.

## Memory budget on a 3090

Approximate peak VRAM (measured by `python -m textsam.train.stage1
--config configs/stage1_phrasecut.yaml --profile-vram-only`):

| Resolution | Batch | Queries/image | Grad checkpoint? | Peak VRAM |
|---|---|---|---|---|
| 1024² | 4 | 1 | no | ~20 GB |
| 1024² | 8 | 1 | yes (decoder) | ~22 GB |
| 512² | 16 | 1 | no | ~14 GB |
| 512² | 4 | 8 | yes (decoder) | ~21 GB |

If you hit OOM, the knobs in order of effectiveness are:

1. lower resolution (1024 → 512 ≈ 4× activation savings)
2. drop K (8 → 4 queries per image)
3. enable `train.grad_checkpointing: true`
4. drop `train.batch_size` and bump `train.grad_accum_steps` to compensate

## Curriculum rationale

Why single-object → multi-object?

- **Stage 1 is a clean optimisation target.** Each gradient step has one
  prediction matched to one ground-truth mask. The cross-modal adapter sees
  every text query as load-bearing (no negative classes to confuse it), and
  the decoder finetunes smoothly from pretrained SAM weights.
- **Stage 2 introduces multiplicity gradually.** By warm-starting from a
  decoder that already knows how to turn a text-conditioned prompt into a
  clean mask, the multi-query loss converges much faster than from scratch.
- This mirrors the curriculum used by LISA [@lai2024lisa] and X-Decoder
  [@zou2023xdecoder], who similarly pretrain on referring-expression
  segmentation before adding multi-class objectives.

## Sanity checks

1. **Self-test the model:**
   ```
   python -m textsam.models.textsam --self-test
   ```
   Builds the model, runs one forward on a 1024² zero image with text
   `"a cat"`, asserts the output mask shape is `(1, 1, 1024, 1024)` and
   contains no NaNs.

2. **Smoke-test the data pipeline:**
   ```
   python -m textsam.data.download --dataset phrasecut --limit 50
   python -m textsam.data.prepare --stage 1
   python -m textsam.data.prepare --check
   ```

3. **Overfit 16 samples:**
   Train for 200 steps on a 16-sample subset and confirm train mIoU > 0.85.
   Confirms loss/optimizer/data wiring are correct before kicking off the
   long stage-1 run.

## References

See `docs/citations.bib`.
