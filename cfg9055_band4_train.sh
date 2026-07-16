#!/usr/bin/env bash
# Self-terminating launcher for the 90/5/5 CFG-branch + Band-4 campaign.
set -euo pipefail

usage() {
  echo "usage: cfg9055_band4_train.sh main10k|smoke300" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

mode=$1
case "$mode" in
  main10k)
    config=examples/config/train_config_cfg9055_band4_10k.json
    default_gpus=4
    ;;
  smoke300)
    config=examples/config/train_config_cfg9055_band4_smoke300.json
    default_gpus=2
    ;;
  *)
    usage
    exit 2
    ;;
esac

run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID contains unsupported characters: $run_id" >&2
  exit 2
fi
num_gpus=${NUM_GPUS:-$default_gpus}
if [[ ! "$num_gpus" =~ ^[1-8]$ ]]; then
  echo "NUM_GPUS must be an integer from 1 through 8: $num_gpus" >&2
  exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
expected_root=${EXPECTED_SOURCE_ROOT:-/opt/gpfs/users/shuai/work/cfg-band4-pretrain-20260717/OmniVoice}
if [[ $(realpath "$root") != $(realpath "$expected_root") ]]; then
  echo "refusing unexpected source root: root=$root expected=$expected_root" >&2
  exit 1
fi
cd "$root"

if [[ -n $(git status --porcelain) ]]; then
  echo "refusing dirty source tree" >&2
  git status --short >&2
  exit 1
fi
source_commit=$(git rev-parse HEAD)

runtime_venv=${RUNTIME_VENV:-/opt/gpfs/users/yinfeng/work/OmniVoice/.venv}
source "$runtime_venv/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$root:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

python - "$root" <<'PY'
import pathlib
import sys
import omnivoice

expected = pathlib.Path(sys.argv[1]).resolve()
actual = pathlib.Path(omnivoice.__file__).resolve()
print(f"python={sys.executable}")
print(f"omnivoice={actual}")
if expected not in actual.parents:
    raise SystemExit(f"refusing non-Shuai OmniVoice source: {actual}")
PY

python scripts/check_cfg9055_training_contract.py
python scripts/check_checkpoint_vocab.py --train-config "$config"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

output_root=${OUTPUT_ROOT:-/opt/gpfs/users/shuai/experiments/cfg-band4-9055-20260717}
output="$output_root/${mode}_${run_id}"
log_dir="$output_root/logs"
log_path="$log_dir/${mode}_${run_id}.log"
manifest="$output_root/${mode}_${run_id}.manifest.txt"
mkdir -p "$log_dir"
if [[ -e "$output" || -e "$log_path" || -e "$manifest" ]]; then
  echo "refusing to reuse artifacts for run_id=$run_id" >&2
  exit 1
fi

{
  echo "run_id=$run_id"
  echo "mode=$mode"
  echo "source_root=$root"
  echo "source_commit=$source_commit"
  echo "source_author=$(git show -s --format='%an <%ae>' HEAD)"
  echo "config=$config"
  echo "config_sha256=$(sha256sum "$config" | awk '{print $1}')"
  echo "data_config_sha256=$(sha256sum examples/config/data_config_emilia_full_blockparity.json | awk '{print $1}')"
  echo "checkpoint=/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice/exp/blockcausal_splitloss_emilia_300k_lx20/checkpoint-300000"
  echo "checkpoint_model_sha256=d72a01f60f01e2a6432779993981d2bdd2100266bdffc0a46a3cb957bf8e1908"
  echo "num_gpus=$num_gpus"
  echo "output=$output"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$manifest"

gpu_ids=$(seq -s, 0 $((num_gpus - 1)))
echo "CFG9055_START mode=$mode run_id=$run_id gpus=$num_gpus commit=$source_commit output=$output time=$(date -u +%FT%TZ)"
set +e
accelerate launch --gpu_ids "$gpu_ids" --num_processes "$num_gpus" \
  -m omnivoice.cli.train \
  --train_config "$config" \
  --data_config examples/config/data_config_emilia_full_blockparity.json \
  --output_dir "$output" \
  2>&1 | tee "$log_path"
rc=${PIPESTATUS[0]}
set -e
echo "CFG9055_EXIT mode=$mode run_id=$run_id rc=$rc time=$(date -u +%FT%TZ)" | tee -a "$log_path"
exit "$rc"
