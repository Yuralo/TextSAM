#!/usr/bin/env bash
# Render dataset previews under viz/:
#   viz/samples_train.png  - random training samples with mask overlays
#   viz/samples_val.png    - same for val
#   viz/transforms.png     - raw / augmented / SAM-preprocessed per row
#   viz/stats.png          - phrase length, phrases-per-image, top phrases, summary
set -euo pipefail
python -m textsam.data.visualize \
    --manifest "${MANIFEST:-datasets/manifest_stage1.jsonl}" \
    --output   "${OUTPUT:-viz}" \
    "$@"
