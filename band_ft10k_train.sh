#!/usr/bin/env bash
# Self-terminating single-node H100 launcher for the EOS-band controlled pair.
set -euo pipefail

arm=${1:?usage: band_ft10k_train.sh band4|bandctl}
case "$arm" in
  band4)
    config=examples/config/train_config_ft10k_band4.json
    output=exp/splitloss_ft10k_eos_band4
    ;;
  bandctl)
    config=examples/config/train_config_ft10k_bandctl.json
    output=exp/splitloss_ft10k_eos_bandctl
    ;;
  *)
    echo "unknown arm: $arm" >&2
    exit 2
    ;;
esac

L=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/OmniVoice
SOURCE_EXP=/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice/exp/blockcausal_splitloss_emilia_300k_lx20
DL=/opt/gpfs/users/yinfeng/work/OmniVoice

cd "$L"
source "$DL/.venv/bin/activate"
export PYTHONPATH="$L:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p exp logs
ln -sfn "$SOURCE_EXP" exp/blockcausal_splitloss_emilia_300k_lx20

echo "BAND_FT_START arm=$arm commit=$(git rev-parse HEAD) time=$(date -u +%FT%TZ)"
accelerate launch --gpu_ids "$(seq -s, 0 7)" --num_processes 8 \
  -m omnivoice.cli.train \
  --train_config "$config" \
  --data_config examples/config/data_config_emilia_full_blockparity.json \
  --output_dir "$output" \
  2>&1 | tee "logs/${arm}_ft10k.log"
echo "BAND_FT_DONE arm=$arm time=$(date -u +%FT%TZ)"
