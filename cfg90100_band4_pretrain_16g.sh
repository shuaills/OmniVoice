#!/usr/bin/env bash
# Self-terminating two-node (2 x 8 H100) launcher for shared10 CFG + Band-4.
set -euo pipefail

usage() {
  echo "usage: cfg90100_band4_pretrain_16g.sh smoke300|perf300" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

mode=$1
case "$mode" in
  smoke300)
    config=examples/config/train_config_cfg90100_band4_smoke300_16g.json
    expected_grad_checkpoint=true
    expected_balanced_packing=0
    ;;
  perf300)
    config=examples/config/train_config_cfg90100_band4_perf300_16g.json
    expected_grad_checkpoint=false
    expected_balanced_packing=0
    ;;
  *)
    usage
    exit 2
    ;;
esac
expected_num_machines=2
expected_gpus_per_machine=8
expected_world_size=16
expected_final_step=300
reference_world_size=8
reference_batch_tokens=15648
reference_gradient_accumulation_steps=1

run_id=${RUN_ID:?RUN_ID is required and must be identical on both worker pods}
if [[ ! $run_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID contains unsupported characters: $run_id" >&2
  exit 2
fi

num_machines=${NUM_MACHINES:-$expected_num_machines}
gpus_per_machine=${GPUS_PER_MACHINE:-$expected_gpus_per_machine}
world_size=$((num_machines * gpus_per_machine))
if [[ $num_machines -ne $expected_num_machines || $gpus_per_machine -ne $expected_gpus_per_machine || $world_size -ne $expected_world_size ]]; then
  echo "refusing topology drift: machines=$num_machines gpus_per_machine=$gpus_per_machine world_size=$world_size" >&2
  exit 2
fi

host=$(hostname)
if [[ $host =~ -worker-([0-9]+)$ ]]; then
  node_rank=${BASH_REMATCH[1]}
else
  echo "cannot derive OMS node_rank from worker hostname: $host" >&2
  exit 2
fi
if [[ $node_rank -lt 0 || $node_rank -ge $num_machines ]]; then
  echo "invalid node_rank=$node_rank for num_machines=$num_machines host=$host" >&2
  exit 2
fi
if [[ ${VC_WORKER_NUM:-} != "$num_machines" ]]; then
  echo "VC_WORKER_NUM mismatch: expected=$num_machines actual=${VC_WORKER_NUM:-unset}" >&2
  exit 2
fi
IFS=, read -r -a worker_hosts <<< "${VC_WORKER_HOSTS:-}"
if [[ ${#worker_hosts[@]} -ne $num_machines ]]; then
  echo "VC_WORKER_HOSTS count mismatch: ${VC_WORKER_HOSTS:-unset}" >&2
  exit 2
fi
job_name=${host%-worker-*}
expected_self="$host.$job_name"
if [[ ${worker_hosts[$node_rank]} != "$expected_self" ]]; then
  echo "worker host/rank mismatch: rank=$node_rank expected=$expected_self actual=${worker_hosts[$node_rank]}" >&2
  exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
expected_root=${EXPECTED_SOURCE_ROOT:-/opt/gpfs/users/shuai/work/cfg90100-band4-16g-smoke-20260718/OmniVoice}
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
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
nccl_debug=${NCCL_DEBUG:-INFO}
if [[ $nccl_debug != INFO || $NCCL_IB_DISABLE != 0 ]]; then
  echo "refusing unverifiable NCCL contract: NCCL_DEBUG=$nccl_debug NCCL_IB_DISABLE=$NCCL_IB_DISABLE" >&2
  exit 1
fi
export NCCL_DEBUG=$nccl_debug

python - "$config" "$world_size" "$reference_world_size" "$reference_batch_tokens" "$reference_gradient_accumulation_steps" "$expected_grad_checkpoint" "$expected_balanced_packing" <<'PY'
import json
import math
import sys

from omnivoice.training.config import TrainingConfig

(
    config_path,
    world_size_raw,
    reference_world_size_raw,
    reference_batch_tokens_raw,
    reference_ga_raw,
    expected_grad_checkpoint_raw,
    expected_balanced_packing_raw,
) = sys.argv[1:]
world_size = int(world_size_raw)
reference_world_size = int(reference_world_size_raw)
reference_batch_tokens = int(reference_batch_tokens_raw)
reference_ga = int(reference_ga_raw)
expected_grad_checkpoint = expected_grad_checkpoint_raw == "true"
expected_balanced_packing = int(expected_balanced_packing_raw)

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
    "batch_tokens": 7824,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "seed": 42,
    "block_size": 32,
    "block_scheme": "dual",
    "steps": 300,
    "save_steps": 300,
    "warmup_type": "steps",
    "warmup_ratio": 0.0,
    "warmup_steps": 9000,
    "perf_grad_checkpoint": expected_grad_checkpoint,
    "perf_balanced_packing": expected_balanced_packing,
}
for key, value in expected.items():
    actual = config.get(key)
    if isinstance(value, float):
        ok = isinstance(actual, (int, float)) and math.isclose(actual, value)
    else:
        ok = actual == value
    if not ok:
        raise SystemExit(f"contract mismatch: {key} expected={value!r} actual={actual!r}")

global_batch_tokens = world_size * config["batch_tokens"] * config["gradient_accumulation_steps"]
reference_global_batch_tokens = reference_world_size * reference_batch_tokens * reference_ga
if global_batch_tokens != reference_global_batch_tokens:
    raise SystemExit(
        "global batch mismatch: "
        f"16g={global_batch_tokens} reference8g={reference_global_batch_tokens}"
    )
print(
    "CFG90100_16G_CONFIG_OK "
    f"world_size={world_size} batch_tokens={config['batch_tokens']} "
    f"ga={config['gradient_accumulation_steps']} global_batch_tokens={global_batch_tokens}"
)
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
if [[ ! -d /sys/class/net/$NCCL_SOCKET_IFNAME ]]; then
  echo "missing NCCL network interface: $NCCL_SOCKET_IFNAME" >&2
  exit 1
fi

gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
gpu_count=$(printf '%s\n' "$gpu_names" | sed '/^$/d' | wc -l)
if [[ $gpu_count -ne $gpus_per_machine ]]; then
  echo "GPU visibility mismatch: expected=$gpus_per_machine actual=$gpu_count" >&2
  exit 1
fi
if printf '%s\n' "$gpu_names" | grep -vi 'H100' >/dev/null; then
  echo "refusing non-H100 GPU inventory:" >&2
  printf '%s\n' "$gpu_names" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

output_root=${OUTPUT_ROOT:-/opt/gpfs/users/shuai/experiments/cfg90100-band4-pretrain16g-20260718}
output="$output_root/${mode}_${run_id}"
log_dir="$output_root/logs"
log_path="$log_dir/${mode}_${run_id}.node${node_rank}.log"
manifest="$output_root/${mode}_${run_id}.manifest.txt"
pass_file="$output_root/${mode}_${run_id}.PASS"
control_root="$output_root/.control"
control_dir="$control_root/$run_id"
meta_file="$control_dir/meta.txt"
master_ip_file="$control_dir/master_ip.txt"
mkdir -p "$output_root" "$log_dir" "$control_root"

die() {
  local message=$1
  echo "$message" >&2
  if [[ -d ${control_dir:-} ]]; then
    printf 'node_rank=%s\nmessage=%s\ntime=%s\n' "$node_rank" "$message" "$(date -u +%FT%TZ)" \
      > "$control_dir/failure.node${node_rank}.tmp"
    mv "$control_dir/failure.node${node_rank}.tmp" "$control_dir/failure.node${node_rank}"
  fi
  exit 1
}

wait_for_file() {
  local path=$1
  local label=$2
  local timeout_seconds=${3:-240}
  local deadline=$((SECONDS + timeout_seconds))
  while [[ ! -s $path ]]; do
    if compgen -G "$control_dir/failure.node*" >/dev/null; then
      die "peer failed while waiting for $label"
    fi
    if ((SECONDS >= deadline)); then
      die "timed out after ${timeout_seconds}s waiting for $label: $path"
    fi
    sleep 2
  done
}

config_sha256=$(sha256sum "$config" | awk '{print $1}')
data_config=examples/config/data_config_emilia_full_blockparity.json
data_config_sha256=$(sha256sum "$data_config" | awk '{print $1}')
baseline_log=${BASELINE_LOG:-/opt/gpfs/users/shuai/experiments/cfg90100-band4-pretrain16g-20260718/logs/smoke300_shuai-cfg90100-band4-smoke300-h16-rdma-v2.node0.log}
baseline_log_sha256=not-applicable
if [[ $mode == perf300 ]]; then
  if [[ ! -s $baseline_log ]]; then
    die "missing frozen 16-GPU RDMA baseline log: $baseline_log"
  fi
  baseline_log_sha256=$(sha256sum "$baseline_log" | awk '{print $1}')
fi
main_process_port=${MASTER_PORT:-29517}
if [[ ! $main_process_port =~ ^[0-9]+$ || $main_process_port -lt 1024 || $main_process_port -gt 65535 ]]; then
  die "invalid MASTER_PORT=$main_process_port"
fi

if [[ $node_rank -eq 0 ]]; then
  for artifact in "$output" "$manifest" "$pass_file" \
    "$log_dir/${mode}_${run_id}.node0.log" "$log_dir/${mode}_${run_id}.node1.log" \
    "$output_root/${mode}_${run_id}.SUMMARY.log" \
    "$output_root/${mode}_${run_id}.SUMMARY.json" \
    "$output_root/${mode}_${run_id}.SUMMARY.tsv" \
    "$output_root/${mode}_${run_id}.VERDICT.txt" \
    "$control_dir"; do
    if [[ -e $artifact ]]; then
      echo "refusing to reuse artifact: $artifact" >&2
      exit 1
    fi
  done
  for rank in 0 1; do
    gpu_artifact="$log_dir/${mode}_${run_id}.node${rank}.gpu.csv"
    if [[ -e $gpu_artifact ]]; then
      echo "refusing to reuse artifact: $gpu_artifact" >&2
      exit 1
    fi
  done
  mkdir "$control_dir"
  master_ip=$(hostname -i | awk '{print $1}')
  if [[ ! $master_ip =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    die "refusing invalid rendezvous IP from hostname -i: $master_ip"
  fi
  created_epoch=$(date +%s)
  {
    echo "run_id=$run_id"
    echo "created_epoch=$created_epoch"
    echo "source_commit=$source_commit"
    echo "config_sha256=$config_sha256"
    echo "baseline_log_sha256=$baseline_log_sha256"
    echo "num_machines=$num_machines"
    echo "gpus_per_machine=$gpus_per_machine"
    echo "world_size=$world_size"
    echo "main_process_port=$main_process_port"
  } > "$meta_file.tmp"
  mv "$meta_file.tmp" "$meta_file"
  printf '%s\n' "$master_ip" > "$master_ip_file.tmp"
  mv "$master_ip_file.tmp" "$master_ip_file"
  {
    echo "run_id=$run_id"
    echo "mode=$mode"
    echo "source_root=$root"
    echo "source_commit=$source_commit"
    echo "source_author=$(git show -s --format='%an <%ae>' HEAD)"
    echo "config=$config"
    echo "config_sha256=$config_sha256"
    echo "data_config=$data_config"
    echo "data_config_sha256=$data_config_sha256"
    echo "baseline_log=$baseline_log"
    echo "baseline_log_sha256=$baseline_log_sha256"
    echo "llm_name_or_path=/opt/gpfs/models/Qwen3-0.6B"
    echo "num_machines=$num_machines"
    echo "gpus_per_machine=$gpus_per_machine"
    echo "world_size=$world_size"
    echo "rank_axes=node_rank:[0,1],local_rank:[0,7],global_rank=node_rank*8+local_rank"
    echo "batch_tokens_per_gpu=7824"
    echo "gradient_accumulation_steps=1"
    echo "global_batch_tokens=125184"
    echo "perf_grad_checkpoint=$expected_grad_checkpoint"
    echo "perf_balanced_packing=$expected_balanced_packing"
    echo "main_process_ip=$master_ip"
    echo "main_process_port=$main_process_port"
    echo "nccl_socket_ifname=$NCCL_SOCKET_IFNAME"
    echo "output=$output"
    echo "started_utc=$(date -u +%FT%TZ)"
  } > "$manifest"
else
  wait_for_file "$meta_file" "rank-0 metadata"
  created_epoch=$(awk -F= '$1 == "created_epoch" {print $2}' "$meta_file")
  meta_run_id=$(awk -F= '$1 == "run_id" {print $2}' "$meta_file")
  meta_source_commit=$(awk -F= '$1 == "source_commit" {print $2}' "$meta_file")
  meta_config_sha256=$(awk -F= '$1 == "config_sha256" {print $2}' "$meta_file")
  meta_baseline_log_sha256=$(awk -F= '$1 == "baseline_log_sha256" {print $2}' "$meta_file")
  now_epoch=$(date +%s)
  age_seconds=$((now_epoch - created_epoch))
  if [[ $meta_run_id != "$run_id" || $meta_source_commit != "$source_commit" || $meta_config_sha256 != "$config_sha256" || $meta_baseline_log_sha256 != "$baseline_log_sha256" ]]; then
    die "rank-0 metadata mismatch"
  fi
  if [[ $age_seconds -lt 0 || $age_seconds -gt 600 ]]; then
    die "stale rank-0 metadata: age_seconds=$age_seconds"
  fi
fi

wait_for_file "$master_ip_file" "master IP"
master_ip=$(<"$master_ip_file")
if [[ ! $master_ip =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  die "invalid master IP in rendezvous file: $master_ip"
fi

{
  echo "host=$host"
  echo "node_rank=$node_rank"
  echo "local_rank_range=0-7"
  echo "global_rank_range=$((node_rank * gpus_per_machine))-$((node_rank * gpus_per_machine + gpus_per_machine - 1))"
  echo "world_size=$world_size"
  echo "time=$(date -u +%FT%TZ)"
} > "$control_dir/preflight.node${node_rank}.tmp"
mv "$control_dir/preflight.node${node_rank}.tmp" "$control_dir/preflight.node${node_rank}.ok"
wait_for_file "$control_dir/preflight.node0.ok" "node-0 preflight"
wait_for_file "$control_dir/preflight.node1.ok" "node-1 preflight"

gpu_ids=$(seq -s, 0 $((gpus_per_machine - 1)))
gpu_csv="$log_dir/${mode}_${run_id}.node${node_rank}.gpu.csv"
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
) > "$gpu_csv" &
monitor_pid=$!
echo "CFG90100_16G_START mode=$mode run_id=$run_id host=$host node_rank=$node_rank local_ranks=0-7 global_ranks=$((node_rank * 8))-$((node_rank * 8 + 7)) world_size=$world_size output=$output time=$(date -u +%FT%TZ)"
set +e
accelerate launch \
  --gpu_ids "$gpu_ids" \
  --num_processes "$world_size" \
  --num_machines "$num_machines" \
  --machine_rank "$node_rank" \
  --main_process_ip "$master_ip" \
  --main_process_port "$main_process_port" \
  --rdzv_backend static \
  --max_restarts 0 \
  -m omnivoice.cli.train \
  --train_config "$config" \
  --data_config "$data_config" \
  --output_dir "$output" \
  2>&1 | tee "$log_path"
pipeline_status=("${PIPESTATUS[@]}")
train_rc=${pipeline_status[0]}
tee_rc=${pipeline_status[1]}
node_rc=$train_rc
if [[ $node_rc -eq 0 && $tee_rc -ne 0 ]]; then
  node_rc=$tee_rc
fi
set -e
cleanup_monitor

if ! grep -F "PERF: gradient checkpointing active=$( [[ $expected_grad_checkpoint == true ]] && echo True || echo False )" "$log_path" >/dev/null; then
  echo "missing runtime gradient-checkpointing assertion in $log_path" >&2
  node_rc=1
fi
if ! grep -F "PERF: balanced packing window=$expected_balanced_packing" "$log_path" >/dev/null; then
  echo "missing runtime balanced-packing assertion in $log_path" >&2
  node_rc=1
fi
if ! grep -F "Using network IB" "$log_path" >/dev/null; then
  echo "missing positive NCCL IB selection proof in $log_path; refusing socket fallback" >&2
  node_rc=1
fi
gdrdma_ids=$( { grep -oE 'NET/IB/[0-7]/GDRDMA' "$log_path" || true; } | sort -u | wc -l)
if [[ $gdrdma_ids -ne 8 ]]; then
  echo "incomplete NCCL GDRDMA proof in $log_path: expected=8 actual=$gdrdma_ids" >&2
  node_rc=1
fi
if grep -Ei 'nan|(^|[^[:alpha:]])inf([^[:alpha:]]|$)|OutOfMemory|OOMKilled|Traceback|ChildFailedError|ProcessExitedException|NCCL[^[:alnum:]]+(error|failed)' "$log_path" >/dev/null; then
  echo "failure or non-finite signature found in $log_path" >&2
  node_rc=1
fi

{
  echo "node_rank=$node_rank"
  echo "host=$host"
  echo "train_rc=$train_rc"
  echo "tee_rc=$tee_rc"
  echo "rc=$node_rc"
  echo "finished_utc=$(date -u +%FT%TZ)"
} > "$control_dir/node${node_rank}.rc.tmp"
mv "$control_dir/node${node_rank}.rc.tmp" "$control_dir/node${node_rank}.rc"

if [[ $node_rank -ne 0 ]]; then
  echo "CFG90100_16G_NODE_EXIT node_rank=$node_rank rc=$node_rc time=$(date -u +%FT%TZ)"
  exit "$node_rc"
fi

wait_for_file "$control_dir/node1.rc" "node-1 exit receipt" 300
peer_rc=$(awk -F= '$1 == "rc" {print $2}' "$control_dir/node1.rc")
rc=$node_rc
if [[ $rc -eq 0 && $peer_rc -ne 0 ]]; then
  rc=$peer_rc
fi

max_memory_mib=0
for rank in 0 1; do
  rank_gpu_csv="$log_dir/${mode}_${run_id}.node${rank}.gpu.csv"
  if [[ ! -s $rank_gpu_csv ]]; then
    echo "missing GPU telemetry: $rank_gpu_csv" | tee -a "$log_path" >&2
    rc=1
    continue
  fi
  rank_rows=$(awk -F, 'NR > 1 && $2 ~ /^[0-9]+$/ {count++} END {print count + 0}' "$rank_gpu_csv")
  rank_gpu_ids=$(awk -F, 'NR > 1 && $2 ~ /^[0-9]+$/ {seen[$2]=1} END {for (id in seen) count++; print count + 0}' "$rank_gpu_csv")
  if ((rank_rows < 8 || rank_gpu_ids != 8)); then
    echo "invalid GPU telemetry: rank=$rank rows=$rank_rows gpu_ids=$rank_gpu_ids path=$rank_gpu_csv" | tee -a "$log_path" >&2
    rc=1
  fi
  rank_max_memory=$(awk -F, 'NR > 1 {if ($3 + 0 > max) max=$3 + 0} END {print max + 0}' "$rank_gpu_csv")
  if ((rank_max_memory > max_memory_mib)); then
    max_memory_mib=$rank_max_memory
  fi
done
if ((max_memory_mib >= 70 * 1024)); then
  echo "peak-memory gate failed: max_memory_mib=$max_memory_mib" | tee -a "$log_path" >&2
  rc=1
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
  if [[ $random_state_count -ne $world_size ]]; then
    echo "incomplete checkpoint: expected $world_size random states, found $random_state_count" | tee -a "$log_path" >&2
    rc=1
  fi
  for global_rank in $(seq 0 $((world_size - 1))); do
    if [[ ! -s $checkpoint/random_states_${global_rank}.pkl ]]; then
      echo "incomplete checkpoint: missing random_states_${global_rank}.pkl" | tee -a "$log_path" >&2
      rc=1
    fi
  done
  model_bytes=$(stat -c %s "$checkpoint/model.safetensors" 2>/dev/null || echo 0)
  optimizer_bytes=$(stat -c %s "$checkpoint/optimizer.bin" 2>/dev/null || echo 0)
  if [[ $model_bytes -lt 2000000000 || $optimizer_bytes -lt 4000000000 ]]; then
    echo "checkpoint size gate failed: model_bytes=$model_bytes optimizer_bytes=$optimizer_bytes" | tee -a "$log_path" >&2
    rc=1
  fi
  if ! python - "$checkpoint/scheduler.bin" <<'PY'
import math
import sys

import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected = {
    "last_epoch": 300,
    "_step_count": 301,
    "base_lrs": [0.0001],
    "_last_lr": [0.0001 * 300 / 9000],
}
for key, value in expected.items():
    actual = state.get(key)
    if isinstance(value, list):
        ok = (
            isinstance(actual, list)
            and len(actual) == len(value)
            and all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(actual, value))
        )
    else:
        ok = actual == value
    if not ok:
        raise SystemExit(f"scheduler contract mismatch: {key} expected={value!r} actual={actual!r}")
print(f"CFG90100_16G_SCHEDULER_OK state={expected}")
PY
  then
    echo "checkpoint scheduler gate failed: $checkpoint/scheduler.bin" | tee -a "$log_path" >&2
    rc=1
  fi
fi

summary_verdict=not-applicable
summary_path=not-applicable
if [[ $rc -eq 0 && $mode == perf300 ]]; then
  summary_prefix="$output_root/${mode}_${run_id}"
  summary_log="$summary_prefix.SUMMARY.log"
  set +e
  python scripts/summarize_cfg90100_perf16.py \
    --baseline-log "$baseline_log" \
    --candidate-log "$log_dir/${mode}_${run_id}.node0.log" \
    --gpu-csv "$log_dir/${mode}_${run_id}.node0.gpu.csv" \
    --gpu-csv "$log_dir/${mode}_${run_id}.node1.gpu.csv" \
    --output-prefix "$summary_prefix" \
    2>&1 | tee "$summary_log"
  summary_pipeline=("${PIPESTATUS[@]}")
  set -e
  if [[ ${summary_pipeline[0]} -ne 0 || ${summary_pipeline[1]} -ne 0 ]]; then
    echo "16-GPU performance summary failed: python_rc=${summary_pipeline[0]} tee_rc=${summary_pipeline[1]}" | tee -a "$log_path" >&2
    rc=1
  else
    summary_path="$summary_prefix.SUMMARY.json"
    summary_verdict=$(awk -F= '$1 == "VERDICT" {print $2}' "$summary_prefix.VERDICT.txt")
    if [[ $summary_verdict != PASS ]]; then
      echo "16-GPU performance gate failed: verdict=$summary_verdict" | tee -a "$log_path" >&2
      rc=1
    fi
  fi
fi

{
  echo "finished_utc=$(date -u +%FT%TZ)"
  echo "node0_rc=$node_rc"
  echo "node1_rc=$peer_rc"
  echo "max_memory_mib=$max_memory_mib"
  echo "summary_verdict=$summary_verdict"
  echo "summary_path=$summary_path"
  echo "rc=$rc"
} >> "$manifest"

if [[ $rc -eq 0 ]]; then
  {
    echo "contract=cfg90100-band4-16g"
    echo "source_commit=$source_commit"
    echo "world_size=$world_size"
    echo "global_batch_tokens=125184"
    echo "perf_grad_checkpoint=$expected_grad_checkpoint"
    echo "perf_balanced_packing=$expected_balanced_packing"
    echo "max_memory_mib=$max_memory_mib"
    echo "summary_verdict=$summary_verdict"
    echo "summary_path=$summary_path"
    echo "checkpoint=$checkpoint"
    echo "manifest=$manifest"
    echo "completed_utc=$(date -u +%FT%TZ)"
  } > "$pass_file.tmp"
  mv "$pass_file.tmp" "$pass_file"
fi

echo "CFG90100_16G_EXIT mode=$mode run_id=$run_id rc=$rc time=$(date -u +%FT%TZ)" | tee -a "$log_path"
exit "$rc"
