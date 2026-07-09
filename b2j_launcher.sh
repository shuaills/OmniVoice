#!/usr/bin/env bash
# B2J: EOS/padding decoupling arm — resume B2G-50k, +8k steps with
# eos_decouple_silence labels (single-column [eos] stop event + real-silence
# void supervision). Root fix for tail residual commit + junction EOS prior.
set -euo pipefail
REPO_ROOT="/opt/gpfs/users/shuai/work/block-b2-eosdecouple/OmniVoice"
DONOR_VENV="/opt/gpfs/users/yinfeng/work/OmniVoice/.venv"
cd "${REPO_ROOT}"
source "${DONOR_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"   # our branch must shadow the venv editable install
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -c "import omnivoice, sys; p=omnivoice.__file__; print(\x27omnivoice from:\x27, p); sys.exit(0 if p.startswith(\x27${REPO_ROOT}\x27) else 1)"
mkdir -p logs
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="$(seq -s, 0 $((NUM_GPUS-1)))"
exec accelerate launch --gpu_ids "${GPU_IDS}" --num_processes "${NUM_GPUS}" \
    -m omnivoice.cli.train \
    --train_config examples/config/train_config_block_b2j_eos_decouple.json \
    --data_config examples/config/data_config_internal_b2g.json \
    --output_dir "${OUTPUT_DIR:-exp/block_b2j_eos_decouple}" 2>&1 | tee -a logs/b2j_train.log
