#!/usr/bin/env bash
# from-scratch qwen3-0.6b + EOS-decouple labels, 500k steps (B2S recipe, labels only change)
# eos_decouple_silence labels (single-column [eos] stop event + real-silence
# void supervision). Root fix for tail residual commit + junction EOS prior.
set -euo pipefail
REPO_ROOT="/opt/gpfs/users/shuai/work/block-b2-eosdecouple/OmniVoice"
DONOR_VENV="/opt/gpfs/users/yinfeng/work/OmniVoice/.venv"
cd "${REPO_ROOT}"
source "${DONOR_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"   # our branch must shadow the venv editable install
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -c "import omnivoice, sys; p=omnivoice.__file__; print('omnivoice from:', p); sys.exit(0 if p.startswith('${REPO_ROOT}') else 1)"
mkdir -p logs
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="$(seq -s, 0 $((NUM_GPUS-1)))"
exec accelerate launch --gpu_ids "${GPU_IDS}" --num_processes "${NUM_GPUS}" \
    -m omnivoice.cli.train \
    --train_config examples/config/train_config_scratch_qwen3-0.6b_eosdecouple_500k.json \
    --data_config examples/config/data_config_internal_b2g.json \
    --output_dir "${OUTPUT_DIR:-exp/scratch_qwen3-0.6b_eosdecouple_500k}" 2>&1 | tee -a logs/scratch_eosdecouple_train.log
