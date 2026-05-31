#!/usr/bin/env bash
# Test the model on your own image, with one or more text queries.
#
# Single query:
#   bash scripts/infer.sh path/to/photo.jpg "dog"
# Multiple queries (uses stage-2 multi-query path):
#   bash scripts/infer.sh path/to/photo.jpg "dog" "person" "ball" "grass"
#
# Env overrides:
#   CKPT=checkpoints/stage1/best.pt   (defaults to stage-2 best)
#   IMAGE_SIZE=1024                   (auto-picks 512 for stage-2 ckpt, 1024 for stage-1)
set -euo pipefail
IMG="${1:?image path required}"
shift
if [ $# -eq 0 ]; then
    echo "at least one word/phrase required" >&2
    exit 1
fi

CKPT="${CKPT:-checkpoints/stage2/best.pt}"
# auto-detect image size from the ckpt path; user can override via env.
case "$CKPT" in
    *stage1*) DEFAULT_IS=1024 ;;
    *)        DEFAULT_IS=512  ;;
esac
IMAGE_SIZE="${IMAGE_SIZE:-$DEFAULT_IS}"

OUT="out/$(basename "${IMG%.*}")"
mkdir -p out

python -m textsam.inference.predict \
    --image "$IMG" \
    --word "$@" \
    --checkpoint "$CKPT" \
    --model-config configs/model.yaml \
    --image-size "$IMAGE_SIZE" \
    --out "$OUT"

echo
echo "Outputs in: out/"
ls -la "$OUT"* 2>/dev/null || true
