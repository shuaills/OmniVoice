#!/usr/bin/env bash
# R3: block-causal + split loss on Emilia, 300k steps.
# Chain: provenance -> 200-step real-run smoke (gate) -> full run -> GUARDIAN
# (sleep infinity holds the 8 GPUs after training ends; stop only by oms job delete).
set -u
L=/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice
cd $L
source /opt/gpfs/users/yinfeng/work/OmniVoice/.venv/bin/activate
export PYTHONPATH="$L:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p logs
{
  echo "R3 provenance: $(date -u +%FT%TZ)"
  echo "git: $(git rev-parse HEAD) dirty=[$(git diff --stat | tail -1)]"
  sha256sum examples/config/train_config_emilia_splitloss_300k.json examples/config/data_config_emilia_full_blockparity.json
  nvidia-smi -L | head -1
  python -c "import torch,accelerate,transformers; print(torch.__version__, accelerate.__version__, transformers.__version__)"
} > logs/r3_provenance.json 2>&1

echo "[stage] 200-step split-loss smoke ($(date +%T))"
accelerate launch --gpu_ids "$(seq -s, 0 7)" --num_processes 8 \
  -m omnivoice.cli.train \
  --train_config examples/config/train_config_emilia_splitloss_smoke.json \
  --data_config examples/config/data_config_emilia_full_blockparity.json \
  --output_dir exp/splitloss_smoke 2>&1 | tee logs/r3_smoke.log | tail -3
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ] || tr "\r" "\n" < logs/r3_smoke.log | grep -aqiE "nan|traceback"; then
  echo "R3_SMOKE_FAILED rc=$rc — holding cards for diagnosis"
  sleep infinity
fi
last=$(tr "\r" "\n" < logs/r3_smoke.log | grep -aoE "loss=[0-9.]+" | tail -1)
echo "R3_SMOKE_PASS rc=$rc $last"

echo "[stage] full 300k split-loss run ($(date +%T))"
accelerate launch --gpu_ids "$(seq -s, 0 7)" --num_processes 8 \
  -m omnivoice.cli.train \
  --train_config examples/config/train_config_emilia_splitloss_300k.json \
  --data_config examples/config/data_config_emilia_full_blockparity.json \
  --output_dir exp/blockcausal_splitloss_emilia_300k 2>&1 | tee -a logs/r3_train.log | tail -2
echo "R3_TRAIN_EXITED rc=${PIPESTATUS[0]} $(date -u +%FT%TZ) — guardian holding cards"
sleep infinity
