#!/usr/bin/env bash
# Evaluate a stage-2 checkpoint on the ADE20K val split.
# Writes metrics.json, iou_hist.png, qualitative.png under eval/stage2/.
#
#   bash scripts/eval_stage2.sh                       # uses checkpoints/stage2/best.pt
#   bash scripts/eval_stage2.sh --limit 200           # smoke test
#   bash scripts/eval_stage2.sh --negative-query-ratio 0.25   # also test refusal of absent classes
#   CKPT=checkpoints/stage2/last.pt bash scripts/eval_stage2.sh
set -euo pipefail
python -m textsam.inference.evaluate_stage2 \
    --checkpoint "${CKPT:-checkpoints/stage2/best.pt}" \
    --model-config configs/model.yaml \
    --stage2-config configs/stage2_ade_lvis.yaml \
    --output "${OUTPUT:-eval/stage2}" \
    "$@"
