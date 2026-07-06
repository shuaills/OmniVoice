#!/usr/bin/env bash
# E1: elastic-canvas fine-tune from the migrated official OmniVoice ckpt.
set -euo pipefail
REPO_ROOT="/opt/gpfs/users/shuai/work/elastic-canvas/OmniVoice"
DONOR_VENV="/opt/gpfs/users/yinfeng/work/OmniVoice/.venv"
cd "${REPO_ROOT}"
source "${DONOR_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}"   # our branch must shadow the venv editable install
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -c "import omnivoice, sys; p=omnivoice.__file__; print(\"omnivoice from:\", p); sys.exit(0 if p.startswith(\"${REPO_ROOT}\") else 1)"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="$(seq -s, 0 $((NUM_GPUS-1)))"
exec accelerate launch --gpu_ids "${GPU_IDS}" --num_processes "${NUM_GPUS}" \
    -m omnivoice.cli.train \
    --train_config examples/config/train_config_elastic_e1.json \
    --data_config examples/config/data_config_emilia_yodas_exp.json \
    --output_dir "${OUTPUT_DIR:-exp/elastic_e1}" 2>&1 | tee -a logs/e1_train.log
