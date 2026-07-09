#!/usr/bin/env bash
set -euo pipefail
REPO=/opt/gpfs/users/shuai/work/block-b2-perf/OmniVoice
source /opt/gpfs/users/shuai/work/block-b2-perf/venv-t29/bin/activate
export PYTHONPATH="${REPO}${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}"
cd "${REPO}"
NUM_GPUS="${NUM_GPUS:-3}"
GPU_IDS="$(seq -s, 0 $((NUM_GPUS-1)))"
TRAIN_CONFIG="${TRAIN_CONFIG:?}"
exec accelerate launch --gpu_ids "${GPU_IDS}" --num_processes "${NUM_GPUS}" \
  -m omnivoice.cli.train \
  --train_config "${TRAIN_CONFIG}" \
  --data_config examples/config/data_config_internal_b2g.json \
  --output_dir "${OUTPUT_DIR:?}"
