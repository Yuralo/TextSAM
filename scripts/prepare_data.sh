#!/usr/bin/env bash
# Build the unified manifests (stage 1: PhraseCut, stage 2: ADE20K + LVIS).
set -euo pipefail

COCO_ROOT="${COCO_ROOT:-/data/coco}"

python -m textsam.data.prepare --stage 1
python -m textsam.data.prepare --stage 2 --coco-root "$COCO_ROOT"
python -m textsam.data.prepare --check
