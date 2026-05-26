#!/usr/bin/env bash
# Usage: bash scripts/infer.sh path/to/image.jpg "a dog" [checkpoint.pt]
set -euo pipefail
IMG="${1:?image path required}"
WORD="${2:?word/phrase required}"
CKPT="${3:-checkpoints/stage2/best.pt}"

python -m textsam.inference.predict \
    --image "$IMG" \
    --word "$WORD" \
    --checkpoint "$CKPT" \
    --model-config configs/model.yaml \
    --out "out/$(basename "${IMG%.*}").png"
