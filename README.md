# TextSAM — Open-Vocabulary Segmentation from `(image, word)`

TextSAM takes an **image** and a **word or short phrase** and produces a
**binary segmentation mask** of the named object inside the image. It is
designed and tuned to train on a single **RTX 3090 (24 GB)**.

```
image  ─►  [frozen SAM ViT-B image enc]  ─┐
                                          ├─►  cross-modal adapter  ─►  SAM mask decoder  ─►  mask
word   ─►  [frozen CLIP ViT-B/16 text enc]┘     (~10 M trainable)        (~4 M, warm-started)
```

The original SAM paper [^sam] described an optional text-prompt slot whose
weights Meta never released. TextSAM realises exactly that missing slot — by
fusing CLIP-encoded text [^clip] with SAM image features through a small
learned adapter and then handing off to SAM's pretrained mask decoder.

## Quickstart

```bash
# 1. Create environment (Python 3.10+) and install
conda create -n textsam python=3.12 -y
conda activate textsam
pip install -r requirements.txt

# 2. Pull datasets + SAM checkpoint   (set COCO_ROOT to your local COCO 2017 root)
export COCO_ROOT=/data/coco
bash scripts/download_data.sh

# 3. Build the unified training manifests
bash scripts/prepare_data.sh

# 4. Stage 1 — single-mask on PhraseCut (~2–3 days on 3090)
bash scripts/train_stage1.sh

# 5. Stage 2 — multi-query on ADE20K + LVIS, warm-started from stage 1 (~3–5 days)
bash scripts/train_stage2.sh

# 6. Use it
bash scripts/infer.sh examples/dog.jpg "dog"
```

## Curriculum

| Stage | Dataset | Input | Output | Why it's first |
|---|---|---|---|---|
| 1 | PhraseCut [^phrasecut] (on Visual Genome [^vg]) | image + phrase | one binary mask | Identical to SAM's native single-prompt → single-mask shape; warm-starts the decoder smoothly. |
| 2 | ADE20K [^ade1; ^ade2] + LVIS [^lvis] (uses your local COCO [^coco] images) | image + K class names | K binary masks | Teaches multi-object handling and refusal of absent classes. |

Each image in both stages has **multiple objects** and **named** masks, per
the user's spec. PhraseCut keeps stage 1 strictly off the COCO image set; LVIS
in stage 2 reuses the COCO images you already have on disk for a 1 200-class
vocabulary at near-zero extra disk cost.

## Architecture (one paragraph)

Two backbones are **frozen** — SAM's ViT-B image encoder and CLIP's
ViT-B/16 text encoder. A small **CrossModalAdapter** projects both into a
shared 256-d space, runs two cross-attention blocks (text attends to image
features, then self-attends with FFN), and pools the refined text tokens into
`K=4` "sparse prompt" tokens that SAM's pretrained MaskDecoder consumes as if
they were point/box prompts. Only the adapter and the mask decoder are
trained — about **10 M parameters total**. Full diagram and design rationale
in [`docs/architecture.md`](docs/architecture.md).

Inspirations and direct prior work:

- **SAM** [^sam] — backbone and mask decoder.
- **CLIP** [^clip] — text encoder.
- **CLIPSeg** [^clipseg] — text-prompted segmentation baseline.
- **Grounded-SAM** [^groundedsam] — closely related zero-shot pipeline (we
  train rather than pipeline).
- **LISA** [^lisa] — referring/reasoning segmentation; we share the
  pretrain-on-refer → finetune-on-multi curriculum.
- **PerSAM** [^persam] — one-shot SAM personalisation; same idea of injecting
  a learned prompt into SAM's decoder.
- **Mask2Former** [^m2f] and **X-Decoder** [^xdec] — loss formulation (Dice +
  Focal + optional Hungarian matcher).

## Repository layout

```
configs/        model + per-stage training configs (YAML)
textsam/
  models/       SAM/CLIP encoders, fusion adapter, decoder, full TextSAM module
  data/         downloaders, manifest builder, per-dataset Dataset classes
  train/        losses, trainer, stage1.py, stage2.py entry points
  inference/    predict CLI + visualization
  utils/        metrics, checkpoint, logging
scripts/        thin shell wrappers around the python entry points
docs/           architecture / datasets / training / inference / citations.bib
```

## Sanity-check the install

```
# Model self-test (random pretrained weights are fine)
python -m textsam.models.textsam --self-test
```

## Hardware notes

- **RTX 3090 (24 GB)**: tested target. Stage 1 fits batch 4 at 1024² in bf16
  with ~20 GB peak. Stage 2 fits batch 4 × K=8 queries at 512² in bf16 with
  gradient checkpointing on the decoder.
- Smaller GPUs: drop to 512² stage 1 and reduce K to 4 in stage 2; both
  trainable on 16 GB.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — full model details and
  trainable-parameter accounting.
- [`docs/datasets.md`](docs/datasets.md) — what each dataset is, how to
  download/prepare, the unified manifest schema.
- [`docs/training.md`](docs/training.md) — curriculum, hyperparameters, VRAM
  budget table, sanity checks.
- [`docs/inference.md`](docs/inference.md) — CLI and Python API examples.
- [`docs/citations.bib`](docs/citations.bib) — BibTeX for every paper
  referenced.

## License

Code: MIT (see `pyproject.toml`).
Datasets follow their respective licenses (see `docs/datasets.md`).

## References

[^sam]: Kirillov et al., *Segment Anything*, ICCV 2023.
[^clip]: Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.
[^clipseg]: Lüddecke and Ecker, *Image Segmentation Using Text and Image Prompts*, CVPR 2022.
[^groundedsam]: Ren et al., *Grounded SAM*, 2024.
[^lisa]: Lai et al., *LISA: Reasoning Segmentation via Large Language Model*, CVPR 2024.
[^persam]: Zhang et al., *Personalize Segment Anything Model with One Shot*, ICLR 2024.
[^m2f]: Cheng et al., *Mask2Former*, CVPR 2022.
[^xdec]: Zou et al., *Generalized Decoding for Pixel, Image and Language*, CVPR 2023.
[^phrasecut]: Wu et al., *PhraseCut: Language-Based Image Segmentation in the Wild*, CVPR 2020.
[^vg]: Krishna et al., *Visual Genome*, IJCV 2017.
[^ade1]: Zhou et al., *Scene Parsing through ADE20K*, CVPR 2017.
[^ade2]: Zhou et al., *Semantic Understanding of Scenes through the ADE20K Dataset*, IJCV 2019.
[^lvis]: Gupta, Dollár, Girshick, *LVIS: A Dataset for Large Vocabulary Instance Segmentation*, CVPR 2019.
[^coco]: Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
