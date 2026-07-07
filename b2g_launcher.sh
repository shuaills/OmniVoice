#!/usr/bin/env bash
# B2G: block conversion on OFFICIAL internal 436kh tokens (zh+en) — native-data form.
set -euo pipefail
REPO_ROOT="/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice"
DONOR_VENV="/opt/gpfs/users/yinfeng/work/OmniVoice/.venv"
cd "${REPO_ROOT}"
source "${DONOR_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}"   # our branch must shadow the venv editable install
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -c "import omnivoice, sys; p=omnivoice.__file__; print('omnivoice from:', p); sys.exit(0 if p.startswith('${REPO_ROOT}') else 1)"
mkdir -p logs
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="$(seq -s, 0 $((NUM_GPUS-1)))"
exec accelerate launch --gpu_ids "${GPU_IDS}" --num_processes "${NUM_GPUS}" \
    -m omnivoice.cli.train \
    --train_config examples/config/train_config_block_b2g.json \
    --data_config examples/config/data_config_internal_b2g.json \
    --output_dir "${OUTPUT_DIR:-exp/block_b2g}" 2>&1 | tee -a logs/b2g_train.log
