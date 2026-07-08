#!/usr/bin/env bash
set -euo pipefail
REPO=/opt/gpfs/users/shuai/work/block-b2-perf/OmniVoice
source /opt/gpfs/users/yinfeng/work/OmniVoice/.venv/bin/activate
export PYTHONPATH="${REPO}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd "${REPO}"
python -c "import omnivoice,sys;p=omnivoice.__file__;print('omnivoice from:',p);sys.exit(0 if p.startswith('${REPO}') else 1)"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="$(seq -s, 0 $((NUM_GPUS-1)))"
exec accelerate launch --gpu_ids "${GPU_IDS}" --num_processes "${NUM_GPUS}" \
  -m omnivoice.cli.train \
  --train_config examples/config/train_config_perf.json \
  --data_config examples/config/data_config_internal_b2g.json \
  --output_dir "${OUTPUT_DIR:?}"
