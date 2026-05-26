#!/usr/bin/env bash
set -euo pipefail
python -m textsam.train.stage1 --config configs/stage1_phrasecut.yaml "$@"
