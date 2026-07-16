#!/usr/bin/env bash
# Self-terminating single-node H100 launcher for the EOS-band controlled pair.
set -euo pipefail

usage() {
  echo "usage: band_ft10k_train.sh band4|bandctl [--smoke]" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

arm=$1
shift
smoke_steps=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      smoke_steps=300
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

case "$arm" in
  band4)
    config=examples/config/train_config_ft10k_band4.json
    output_prefix=exp/splitloss_ft10k_eos_band4
    ;;
  bandctl)
    config=examples/config/train_config_ft10k_bandctl.json
    output_prefix=exp/splitloss_ft10k_eos_bandctl
    ;;
  *)
    echo "unknown arm: $arm" >&2
    exit 2
    ;;
esac

run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID contains unsupported characters: $run_id" >&2
  exit 2
fi
num_gpus=${NUM_GPUS:-8}
if [[ ! "$num_gpus" =~ ^[1-8]$ ]]; then
  echo "NUM_GPUS must be an integer from 1 through 8: $num_gpus" >&2
  exit 2
fi

L=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_EXP=/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice/exp/blockcausal_splitloss_emilia_300k_lx20
DL=/opt/gpfs/users/yinfeng/work/OmniVoice

cd "$L"
source "$DL/.venv/bin/activate"
export PYTHONPATH="$L:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p exp logs
ln -sfn "$SOURCE_EXP" exp/blockcausal_splitloss_emilia_300k_lx20

run_config=$config
log_name=${arm}_ft10k
if [[ -n "$smoke_steps" ]]; then
  output_prefix="${output_prefix}_smoke${smoke_steps}"
  log_name="${log_name}_smoke${smoke_steps}"
  run_config=$(mktemp "${TMPDIR:-/tmp}/omnivoice-${arm}-smoke.XXXXXX.json")
  trap 'rm -f "$run_config"' EXIT
  python - "$config" "$run_config" "$smoke_steps" <<'PY'
import json
import sys

source, destination, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(source) as f:
    config = json.load(f)
config["steps"] = steps
config["save_steps"] = steps
config["keep_last_n_checkpoints"] = 1
with open(destination, "w") as f:
    json.dump(config, f, indent=2)
PY
fi

output="${output_prefix}_${run_id}"
log_path="logs/${log_name}_${run_id}.log"
if [[ -e "$output" || -e "$log_path" ]]; then
  echo "refusing to reuse training artifacts: output=$output log=$log_path" >&2
  exit 1
fi

# Header-only/tokenizer preflight runs once before accelerate allocates ranks.
python scripts/check_checkpoint_vocab.py --train-config "$run_config"

echo "BAND_FT_START arm=$arm steps=${smoke_steps:-10000} run_id=$run_id gpus=$num_gpus output=$output commit=$(git rev-parse HEAD) time=$(date -u +%FT%TZ)"
accelerate launch --gpu_ids "$(seq -s, 0 $((num_gpus - 1)))" --num_processes "$num_gpus" \
  -m omnivoice.cli.train \
  --train_config "$run_config" \
  --data_config examples/config/data_config_emilia_full_blockparity.json \
  --output_dir "$output" \
  2>&1 | tee "$log_path"
echo "BAND_FT_DONE arm=$arm time=$(date -u +%FT%TZ)"
