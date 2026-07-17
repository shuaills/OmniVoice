#!/usr/bin/env bash
# Self-terminating launcher for shared10 CFG + Band-4 from-scratch pretraining.
set -euo pipefail

usage() {
  echo "usage: cfg90100_band4_pretrain.sh smoke300|main300k" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

mode=$1
case "$mode" in
  smoke300)
    config=examples/config/train_config_cfg90100_band4_smoke300.json
    expected_gpus=4
    expected_final_step=300
    ;;
  main300k)
    config=examples/config/train_config_cfg90100_band4_300k.json
    expected_gpus=8
    expected_final_step=300000
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
num_gpus=${NUM_GPUS:-$expected_gpus}
if [[ $num_gpus -ne $expected_gpus ]]; then
  echo "refusing world-size drift: mode=$mode expected_gpus=$expected_gpus actual=$num_gpus" >&2
  exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
expected_root=${EXPECTED_SOURCE_ROOT:-/opt/gpfs/users/shuai/work/cfg90100-band4-pretrain-20260717/OmniVoice}
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
expected_source_commit=${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required}
if [[ $source_commit != "$expected_source_commit" ]]; then
  echo "refusing source commit drift: expected=$expected_source_commit actual=$source_commit" >&2
  exit 1
fi

runtime_venv=${RUNTIME_VENV:-/opt/gpfs/users/yinfeng/work/OmniVoice/.venv}
source "$runtime_venv/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$root:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

python - "$config" "$mode" <<'PY'
import json
import math
import sys

from omnivoice.training.config import TrainingConfig

config_path, mode = sys.argv[1:]
with open(config_path) as stream:
    config = json.load(stream)

unknown_keys = sorted(set(config) - set(TrainingConfig.__annotations__))
if unknown_keys:
    raise SystemExit(f"unknown training config keys: {unknown_keys}")

expected = {
    "cfg_branch_training": True,
    "cfg_branch_cond_ratio": 0.90,
    "cfg_branch_shared_ratio": 0.10,
    "cfg_branch_drop_ref_ratio": 0.0,
    "cfg_branch_seed": 42,
    "eos_band_k": 4,
    "split_loss": True,
    "split_gamma": 0.8718,
    "lambda_eos": 0.03728,
    "lambda_void": 0.14516,
    "init_from_checkpoint": None,
    "resume_from_checkpoint": None,
    "learning_rate": 0.0001,
    "lr_scheduler_type": "cosine",
    "batch_tokens": 15648,
    "mixed_precision": "bf16",
    "seed": 42,
    "block_size": 32,
    "block_scheme": "dual",
}
for key, value in expected.items():
    actual = config.get(key)
    if isinstance(value, float):
        ok = isinstance(actual, (int, float)) and math.isclose(actual, value)
    else:
        ok = actual == value
    if not ok:
        raise SystemExit(f"contract mismatch: {key} expected={value!r} actual={actual!r}")

mode_expected = {
    "smoke300": {
        "steps": 300,
        "save_steps": 300,
        "gradient_accumulation_steps": 2,
        "warmup_type": "steps",
        "warmup_ratio": 0.0,
        "warmup_steps": 9000,
    },
    "main300k": {
        "steps": 300_000,
        "save_steps": 10_000,
        "gradient_accumulation_steps": 1,
        "warmup_type": "ratio",
        "warmup_ratio": 0.03,
        "warmup_steps": 0,
    },
}[mode]
for key, value in mode_expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"mode contract mismatch: mode={mode} {key} expected={value!r} actual={config.get(key)!r}"
        )
print(f"CFG90100_ARM_CONFIG_OK mode={mode} contract={expected} mode_contract={mode_expected}")
PY

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

python scripts/check_cfg90100_pretrain_contract.py
test -d /opt/gpfs/models/Qwen3-0.6B
gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
gpu_count=$(printf '%s\n' "$gpu_names" | sed '/^$/d' | wc -l)
if [[ $gpu_count -ne $num_gpus ]]; then
  echo "GPU visibility mismatch: expected=$num_gpus actual=$gpu_count" >&2
  exit 1
fi
if printf '%s\n' "$gpu_names" | grep -vi 'H100' >/dev/null; then
  echo "refusing non-H100 GPU inventory:" >&2
  printf '%s\n' "$gpu_names" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

output_root=${OUTPUT_ROOT:-/opt/gpfs/users/shuai/experiments/cfg90100-band4-pretrain-20260717}
output="$output_root/${mode}_${run_id}"
log_dir="$output_root/logs"
log_path="$log_dir/${mode}_${run_id}.log"
manifest="$output_root/${mode}_${run_id}.manifest.txt"
mkdir -p "$log_dir"
if [[ -e $output || -e $log_path || -e $manifest ]]; then
  echo "refusing to reuse artifacts for run_id=$run_id" >&2
  exit 1
fi

if [[ $mode == main300k ]]; then
  smoke_pass_file=${SMOKE_PASS_FILE:?SMOKE_PASS_FILE is required for main300k}
  if [[ ! -s $smoke_pass_file ]]; then
    echo "missing smoke PASS receipt: $smoke_pass_file" >&2
    exit 1
  fi
  if ! grep -Fx "source_commit=$source_commit" "$smoke_pass_file" >/dev/null; then
    echo "smoke PASS source commit mismatch: $smoke_pass_file" >&2
    exit 1
  fi
  if ! grep -Fx "contract=cfg90100-band4" "$smoke_pass_file" >/dev/null; then
    echo "smoke PASS contract mismatch: $smoke_pass_file" >&2
    exit 1
  fi
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
  echo "llm_name_or_path=/opt/gpfs/models/Qwen3-0.6B"
  echo "num_gpus=$num_gpus"
  echo "output=$output"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$manifest"

gpu_ids=$(seq -s, 0 $((num_gpus - 1)))
echo "CFG90100_START mode=$mode run_id=$run_id gpus=$num_gpus commit=$source_commit output=$output time=$(date -u +%FT%TZ)"
set +e
accelerate launch --gpu_ids "$gpu_ids" --num_processes "$num_gpus" --num_machines 1 \
  -m omnivoice.cli.train \
  --train_config "$config" \
  --data_config examples/config/data_config_emilia_full_blockparity.json \
  --output_dir "$output" \
  2>&1 | tee "$log_path"
pipeline_status=("${PIPESTATUS[@]}")
train_rc=${pipeline_status[0]}
tee_rc=${pipeline_status[1]}
rc=$train_rc
if [[ $rc -eq 0 && $tee_rc -ne 0 ]]; then
  rc=$tee_rc
fi

checkpoint="$output/checkpoint-$expected_final_step"
if [[ $rc -eq 0 ]]; then
  for required in model.safetensors optimizer.bin scheduler.bin train_config.json tokenizer.json tokenizer_config.json; do
    if [[ ! -s $checkpoint/$required ]]; then
      echo "incomplete checkpoint: missing or empty $checkpoint/$required" | tee -a "$log_path" >&2
      rc=1
    fi
  done
  random_state_count=$(find "$checkpoint" -maxdepth 1 -type f -name 'random_states_*.pkl' -size +0c | wc -l)
  if [[ $random_state_count -ne $num_gpus ]]; then
    echo "incomplete checkpoint: expected $num_gpus random states, found $random_state_count" | tee -a "$log_path" >&2
    rc=1
  fi
  model_bytes=$(stat -c %s "$checkpoint/model.safetensors" 2>/dev/null || echo 0)
  optimizer_bytes=$(stat -c %s "$checkpoint/optimizer.bin" 2>/dev/null || echo 0)
  if [[ $model_bytes -lt 2000000000 || $optimizer_bytes -lt 4000000000 ]]; then
    echo "checkpoint size gate failed: model_bytes=$model_bytes optimizer_bytes=$optimizer_bytes" | tee -a "$log_path" >&2
    rc=1
  fi
fi

if [[ $rc -eq 0 && $mode == smoke300 ]]; then
  smoke_pass_file="$output_root/smoke300_${run_id}.PASS"
  if [[ -e $smoke_pass_file ]]; then
    echo "refusing to overwrite smoke PASS receipt: $smoke_pass_file" | tee -a "$log_path" >&2
    rc=1
  else
    {
      echo "contract=cfg90100-band4"
      echo "source_commit=$source_commit"
      echo "checkpoint=$checkpoint"
      echo "manifest=$manifest"
      echo "completed_utc=$(date -u +%FT%TZ)"
    } > "$smoke_pass_file"
  fi
fi

{
  echo "finished_utc=$(date -u +%FT%TZ)"
  echo "train_rc=$train_rc"
  echo "tee_rc=$tee_rc"
  echo "rc=$rc"
} >> "$manifest"
echo "CFG90100_EXIT mode=$mode run_id=$run_id rc=$rc time=$(date -u +%FT%TZ)" | tee -a "$log_path"
exit "$rc"
