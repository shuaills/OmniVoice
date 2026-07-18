#!/usr/bin/env bash
# Self-terminating, sequential 8xH100 performance matrix for CFG90100 Band-4.
set -euo pipefail

usage() {
  echo "usage: cfg90100_band4_perf8.sh matrix300" >&2
}

if [[ $# -ne 1 || $1 != matrix300 ]]; then
  usage
  exit 2
fi

run_id=${RUN_ID:?RUN_ID is required}
if [[ ! $run_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID contains unsupported characters: $run_id" >&2
  exit 2
fi
expected_gpus=8
num_gpus=${NUM_GPUS:-$expected_gpus}
if [[ $num_gpus -ne $expected_gpus ]]; then
  echo "refusing world-size drift: expected=$expected_gpus actual=$num_gpus" >&2
  exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
expected_root=${EXPECTED_SOURCE_ROOT:?EXPECTED_SOURCE_ROOT is required}
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

REQUIRE_OMNIVOICE_RUNTIME=1 python scripts/test_cfg90100_perf_receipt.py
python scripts/check_cfg90100_pretrain_contract.py
test -d /opt/gpfs/models/Qwen3-0.6B
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

gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
gpu_count=$(printf '%s\n' "$gpu_names" | sed '/^$/d' | wc -l)
if [[ $gpu_count -ne $num_gpus ]]; then
  echo "GPU visibility mismatch: expected=$num_gpus actual=$gpu_count" >&2
  exit 1
fi
if printf '%s\n' "$gpu_names" | grep -vi H100 >/dev/null; then
  echo "refusing non-H100 GPU inventory" >&2
  printf '%s\n' "$gpu_names" >&2
  exit 1
fi

output_root=${OUTPUT_ROOT:-/opt/gpfs/users/shuai/experiments/cfg90100-band4-perf8-20260719}
run_root="$output_root/$run_id"
manifest="$run_root/MANIFEST.txt"
complete_file="$run_root/CFG90100_PERF8_COMPLETE"
pass_file="$run_root/CFG90100_PERF8_PASS"
if [[ -e $run_root ]]; then
  echo "refusing to reuse run root: $run_root" >&2
  exit 1
fi
mkdir -p "$run_root/logs" "$run_root/gpu" "$run_root/checkpoints"

arms=(base nogc nogc_bal8 small_nogc small_nogc_bal8 base_replay)
configs=(
  examples/config/train_config_cfg90100_band4_perf8_base.json
  examples/config/train_config_cfg90100_band4_perf8_nogc.json
  examples/config/train_config_cfg90100_band4_perf8_nogc_bal8.json
  examples/config/train_config_cfg90100_band4_perf8_small_nogc.json
  examples/config/train_config_cfg90100_band4_perf8_small_nogc_bal8.json
  examples/config/train_config_cfg90100_band4_perf8_base.json
)

{
  echo "run_id=$run_id"
  echo "contract=cfg90100-band4-perf8-matrix300"
  echo "source_root=$root"
  echo "source_commit=$source_commit"
  echo "source_author=$(git show -s --format='%an <%ae>' HEAD)"
  echo "num_gpus=$num_gpus"
  echo "arms=${arms[*]}"
  echo "steady_state_window=steps100-290"
  echo "started_utc=$(date -u +%FT%TZ)"
  for index in "${!arms[@]}"; do
    echo "arm.${arms[$index]}.config=${configs[$index]}"
    echo "arm.${arms[$index]}.config_sha256=$(sha256sum "${configs[$index]}" | awk '{print $1}')"
  done
} > "$manifest"

monitor_pid=""
cleanup_monitor() {
  if [[ -n ${monitor_pid:-} ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}
on_signal() {
  trap - EXIT INT TERM
  cleanup_monitor
  running_jobs=$(jobs -pr)
  if [[ -n $running_jobs ]]; then
    kill $running_jobs 2>/dev/null || true
  fi
  exit 143
}
trap cleanup_monitor EXIT
trap on_signal INT TERM

start_gpu_monitor() {
  local csv_path=$1
  (
    echo "timestamp_utc,index,memory_used_mib,utilization_gpu_percent"
    while true; do
      timestamp=$(date -u +%FT%TZ)
      nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F, -v timestamp="$timestamp" '{gsub(/ /, "", $0); print timestamp "," $0}'
      sleep 1
    done
  ) > "$csv_path" &
  monitor_pid=$!
}

validate_checkpoint() {
  local arm=$1
  local checkpoint=$2
  for required in model.safetensors optimizer.bin scheduler.bin train_config.json tokenizer.json tokenizer_config.json; do
    if [[ ! -s $checkpoint/$required ]]; then
      echo "arm=$arm incomplete checkpoint: $checkpoint/$required" >&2
      return 1
    fi
  done
  local random_state_count
  random_state_count=$(find "$checkpoint" -maxdepth 1 -type f -name 'random_states_*.pkl' -size +0c | wc -l)
  if [[ $random_state_count -ne $num_gpus ]]; then
    echo "arm=$arm expected $num_gpus random states, found $random_state_count" >&2
    return 1
  fi
  for rank in $(seq 0 $((num_gpus - 1))); do
    if [[ ! -s $checkpoint/random_states_${rank}.pkl ]]; then
      echo "arm=$arm missing random_states_${rank}.pkl" >&2
      return 1
    fi
  done
}

gpu_ids=$(seq -s, 0 $((num_gpus - 1)))
for index in "${!arms[@]}"; do
  arm=${arms[$index]}
  config=${configs[$index]}
  output="$run_root/checkpoints/$arm"
  log_path="$run_root/logs/$arm.log"
  gpu_csv="$run_root/gpu/$arm.csv"
  echo "CFG90100_PERF8_ARM_START arm=$arm config=$config time=$(date -u +%FT%TZ)" | tee "$log_path"
  start_gpu_monitor "$gpu_csv"
  set +e
  accelerate launch \
    --gpu_ids "$gpu_ids" \
    --num_processes "$num_gpus" \
    --num_machines 1 \
    -m omnivoice.cli.train \
    --train_config "$config" \
    --data_config examples/config/data_config_emilia_full_blockparity.json \
    --output_dir "$output" \
    2>&1 | tee -a "$log_path"
  pipeline_status=("${PIPESTATUS[@]}")
  train_rc=${pipeline_status[0]}
  tee_rc=${pipeline_status[1]}
  set -e
  cleanup_monitor
  if [[ $train_rc -ne 0 || $tee_rc -ne 0 ]]; then
    echo "arm=$arm failed train_rc=$train_rc tee_rc=$tee_rc" | tee -a "$log_path" >&2
    exit 1
  fi
  if grep -Ei 'nan|(^|[^[:alpha:]])inf([^[:alpha:]]|$)|OutOfMemory|OOMKilled|Traceback|ChildFailedError|ProcessExitedException|NCCL[^[:alnum:]]+(error|failed)' "$log_path" >/dev/null; then
    echo "arm=$arm failure signature found in $log_path" >&2
    exit 1
  fi
  expected_gc=True
  expected_balanced=0
  if [[ $arm != base && $arm != base_replay ]]; then
    expected_gc=False
  fi
  if [[ $arm == nogc_bal8 || $arm == small_nogc_bal8 ]]; then
    expected_balanced=8
  fi
  if ! grep -F "PERF: gradient checkpointing active=$expected_gc" "$log_path" >/dev/null; then
    echo "arm=$arm missing runtime gradient-checkpointing assertion: expected=$expected_gc" >&2
    exit 1
  fi
  if ! grep -F "PERF: balanced packing window=$expected_balanced" "$log_path" >/dev/null; then
    echo "arm=$arm missing runtime balanced-packing assertion: expected=$expected_balanced" >&2
    exit 1
  fi
  validate_checkpoint "$arm" "$output/checkpoint-300"
  echo "CFG90100_PERF8_ARM_DONE arm=$arm train_rc=$train_rc tee_rc=$tee_rc time=$(date -u +%FT%TZ)" | tee -a "$log_path"
done

python scripts/summarize_cfg90100_perf8.py "$run_root" | tee "$run_root/SUMMARY.log"
verdict=$(awk -F= '$1 == "VERDICT" {print $2}' "$run_root/VERDICT.txt")
{
  echo "finished_utc=$(date -u +%FT%TZ)"
  echo "verdict=$verdict"
  echo "summary=$run_root/SUMMARY.json"
} >> "$manifest"
{
  echo "run_id=$run_id"
  echo "source_commit=$source_commit"
  echo "verdict=$verdict"
  echo "manifest=$manifest"
  echo "completed_utc=$(date -u +%FT%TZ)"
} > "$complete_file"
if [[ $verdict == PASS ]]; then
  cp "$complete_file" "$pass_file"
fi
echo "CFG90100_PERF8_MATRIX_DONE run_id=$run_id verdict=$verdict time=$(date -u +%FT%TZ)"
