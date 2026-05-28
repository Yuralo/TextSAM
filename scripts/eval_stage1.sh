#!/usr/bin/env bash
# Evaluate a stage-1 checkpoint on the PhraseCut val split.
# Writes metrics.json, iou_hist.png, qualitative.png under eval/stage1/.
#
#   bash scripts/eval_stage1.sh                         # uses checkpoints/stage1/best.pt
#   bash scripts/eval_stage1.sh --checkpoint <path>     # override
#   CKPT=checkpoints/stage1/last.pt bash scripts/eval_stage1.sh
set -euo pipefail
python -m textsam.inference.evaluate \
    --checkpoint "${CKPT:-checkpoints/stage1/best.pt}" \
    --model-config configs/model.yaml \
    --manifest "${MANIFEST:-datasets/manifest_stage1.jsonl}" \
    --output "${OUTPUT:-eval/stage1}" \
    "$@"
