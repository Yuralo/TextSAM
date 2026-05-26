# Datasets

TextSAM trains on three sources, picked because each contains **multi-object
scenes with named masks** and together they cover both the single-object
(stage 1) and multi-object (stage 2) curriculum without overlapping with the
COCO image set the user already has on disk *for stage 1* (stage 2 reuses
those COCO images via LVIS).

## Summary

| Dataset | Images | Categories / Vocab | Annotation type | Stage | Disk |
|---|---|---|---|---|---|
| **PhraseCut** [@wu2020phrasecut] (on Visual Genome [@krishna2017vg]) | ~77 K train / ~11 K val / ~11 K test | open-ended phrases | polygon, single mask per phrase | **1** | ~20 GB (VG images only for referenced IDs) |
| **ADE20K** [@zhou2017ade20k; @zhou2019ade20k] | 20 210 train / 2 000 val | **150 named** classes | dense semantic PNG | **2** | ~3.8 GB |
| **LVIS v1** [@gupta2019lvis] (uses MS-COCO [@lin2014coco] images you already have) | 100 K train / 19.8 K val | **1 203 named** classes (long-tail) | polygons / RLE per instance | **2** | ~1 GB (annotations only — images reused) |
| Total stage-1 named-mask samples | ~108 K | – | – | 1 | – |
| Total stage-2 multi-object images | ~120 K | 1 353 unique class names | – | 2 | – |

## Why these three (and not COCO alone)

The user already has COCO 2017 on disk and asked for *something besides* COCO.
The pipeline therefore:

- For **stage 1** uses **PhraseCut**, which lives on Visual Genome images and
  has no overlap with the COCO image filenames you'd normally see. PhraseCut
  is the canonical referring-expression segmentation dataset and is *the
  closest possible match to the curriculum* — every sample is (image, phrase,
  one mask), which is the exact shape SAM's mask decoder was designed for.
- For **stage 2** uses **ADE20K** (a completely separate scene-parsing dataset
  with 150 named classes) and **LVIS** (1 203 named classes on top of the COCO
  images you already own). LVIS-on-your-COCO is the cheapest path to a
  long-tail vocabulary: it costs ~1 GB of annotation JSON, no image
  re-downloading.

If you want a truly **zero-COCO** pipeline, drop LVIS from the stage-2 config
(`configs/stage2_ade_lvis.yaml`) — ADE20K alone still trains the multi-query
head, just with a smaller vocabulary.

## Download

```
# 1. Set your existing COCO root (needed only for LVIS image paths)
export COCO_ROOT=/data/coco       # contains train2017/ and val2017/

# 2. Pull everything
bash scripts/download_data.sh
# or selectively:
python -m textsam.data.download --dataset sam        # SAM ViT-B checkpoint
python -m textsam.data.download --dataset phrasecut  # VG images referenced by PhraseCut
python -m textsam.data.download --dataset ade20k     # ADE20K scene parsing zip
python -m textsam.data.download --dataset lvis --coco-root $COCO_ROOT
```

On-disk layout after download:

```
datasets/
├── phrasecut/
│   ├── refer_train.json, refer_val.json, refer_test.json
│   └── images/{image_id}.jpg              # Visual Genome JPEGs
├── ade20k/
│   ├── ADEChallengeData2016/
│   │   ├── images/training/*.jpg, validation/*.jpg
│   │   ├── annotations/training/*.png, validation/*.png
│   │   └── objectInfo150.txt
│   └── class_names.txt
└── lvis/
    ├── lvis_v1_train.json, lvis_v1_val.json
    └── images -> $COCO_ROOT              # symlink
```

## Prepare (unified manifests)

```
bash scripts/prepare_data.sh
# or:
python -m textsam.data.prepare --stage 1
python -m textsam.data.prepare --stage 2 --coco-root $COCO_ROOT
python -m textsam.data.prepare --check
```

This writes two JSONL manifests that the PyTorch datasets read:

- `datasets/manifest_stage1.jsonl` — one line per PhraseCut sample:
  ```
  {"image": "...", "polygons": [[x,y,...]], "text": "the red mug",
   "dataset": "phrasecut", "split": "train"}
  ```
- `datasets/manifest_stage2.jsonl` — one line per ADE20K image plus one line
  per LVIS image:
  ```
  {"image": "...", "label_png": "...", "classes_present": [3,47,...],
   "class_names_present": ["sky","road",...], "dataset": "ade20k", "split": "train"}
  {"image": "...", "annotations": [{"category_id": 481,
   "category_name": "tabby cat", "segmentation": <RLE/polys>}, ...],
   "image_h": 480, "image_w": 640, "dataset": "lvis", "split": "train"}
  ```

The `--check` step verifies every referenced file exists and reports counts.

## Licenses

- PhraseCut: CC-BY (annotations); Visual Genome images: CC-BY-4.0.
- ADE20K: BSD; see the dataset's terms of use page.
- LVIS: CC-BY-4.0 (annotations); COCO images: CC-BY-4.0.

## Why each one fits the user's spec

> "the data also needs to be images with multiple objects inside them, and
> also the data should include the name of each object"

- **PhraseCut** — every image is a Visual Genome scene with many objects; each
  sample picks one of them via a referring phrase. ✓ multi-object scenes,
  ✓ name (the phrase itself).
- **ADE20K** — scene parsing dataset; the average image contains 5–10 named
  object classes simultaneously. ✓ multi-object, ✓ named (150 classes).
- **LVIS** — explicitly designed to densify COCO with long-tail categories;
  average ~11 instances/image, 1 203 named classes. ✓ multi-object, ✓ named.

## References

See `docs/citations.bib` for full bibliographic info on each `[@key]`.
