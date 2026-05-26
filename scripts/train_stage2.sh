#!/usr/bin/env bash
set -euo pipefail
python -m textsam.train.stage2 --config configs/stage2_ade_lvis.yaml "$@"
