#!/usr/bin/env bash
# From-scratch 300k @ lambda_eos x20 (0.03728): Qwen3-0.6B init, Emilia full, R2-parity recipe.
# Runs on the endpoint-eval guardian pod. Guardian sleep provided by pod itself.
set -u
L=/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice
DL=/opt/gpfs/users/yinfeng/work/OmniVoice
cd $L
source $DL/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$L:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"
mkdir -p logs
accelerate launch --gpu_ids "$(seq -s, 0 7)" --num_processes 8 \
  -m omnivoice.cli.train \
  --train_config examples/config/train_config_emilia_splitloss_300k_lx20.json \
  --data_config examples/config/data_config_emilia_full_blockparity.json \
  --output_dir exp/blockcausal_splitloss_emilia_300k_lx20 2>&1 | tee logs/splitloss_300k_lx20.log | tail -2
echo "SPLITLOSS_300K_LX20_EXITED rc=${PIPESTATUS[0]} $(date +%F_%T)"
