#!/usr/bin/env bash
# 8-GPU big-batch arm: GA=2 (global batch = 2x paper parity = 16-card equivalent),
# 150k steps (same token budget as 300k@1x), lr sqrt(2)-scaled to 1.41e-4, lambda_eos x20.
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
  --train_config examples/config/train_config_emilia_splitloss_bigbatch_ga2_150k_lx20.json \
  --data_config examples/config/data_config_emilia_full_blockparity.json \
  --output_dir exp/blockcausal_splitloss_emilia_bigbatch_ga2_150k_lx20 2>&1 | tee logs/splitloss_bigbatch_ga2.log | tail -2
echo "BIGBATCH_GA2_EXITED rc=${PIPESTATUS[0]} $(date +%F_%T)"
sleep infinity
