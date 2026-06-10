#!/usr/bin/env bash
# Download SAM checkpoint + PhraseCut + ADE20K + LVIS annotations.
# Point COCO_ROOT at your existing local COCO 2017 directory (containing
# train2017/ and val2017/) so LVIS can symlink images into the right place.
set -euo pipefail

COCO_ROOT="${COCO_ROOT:-/data/coco}"

PHRASECUT_START_FROM="${PHRASECUT_START_FROM:-2299}"

python -m textsam.data.download --dataset sam
python -m textsam.data.download --dataset phrasecut --start-from "$PHRASECUT_START_FROM"
python -m textsam.data.download --dataset ade20k
python -m textsam.data.download --dataset lvis --coco-root "$COCO_ROOT"

echo "All downloads complete. Now run: bash scripts/prepare_data.sh"
