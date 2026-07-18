#!/usr/bin/env bash
# Self-terminating 300-step control/head proof for the DSpark-style local head.
set -euo pipefail

phase=preflight
manifest=""
sampler_pid=""
finished=0

stop_sampler() {
  local mode=${1:-strict}
  local pid=${sampler_pid:-}
  local sampler_rc=0
  local forced_kill=0
  local attempt

  [[ -n $pid ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    set +e
    wait "$pid"
    sampler_rc=$?
    set -e
    sampler_pid=""
    if [[ $mode == strict ]]; then
      echo "GPU memory sampler exited before training arm completed rc=$sampler_rc" >&2
      return 1
    fi
    return 0
  fi

  kill -TERM "$pid" 2>/dev/null || true
  for ((attempt = 0; attempt < 50; attempt++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "GPU memory sampler TERM timeout pid=$pid; forcing KILL" >&2
    kill -KILL "$pid" 2>/dev/null || true
    forced_kill=1
  fi
  set +e
  wait "$pid"
  sampler_rc=$?
  set -e
  sampler_pid=""

  if [[ $mode == strict && $forced_kill -ne 0 ]]; then
    return 1
  fi
  if [[ $mode == strict && $sampler_rc -ne 0 && $sampler_rc -ne 137 && $sampler_rc -ne 143 ]]; then
    echo "GPU memory sampler failed rc=$sampler_rc" >&2
    return 1
  fi
  return 0
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n $sampler_pid ]]; then
    set +e
    stop_sampler cleanup
    set -e
  fi
  if [[ $finished -eq 0 && -n $manifest && -e $manifest ]]; then
    {
      echo "failed_phase=$phase"
      echo "infrastructure_rc=$rc"
      echo "finished_utc=$(date -u +%FT%TZ)"
    } >> "$manifest"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ $# -ne 0 ]]; then
  echo "usage: dspark_markov_300ab.sh" >&2
  exit 2
fi

run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ ! $run_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID contains unsupported characters: $run_id" >&2
  exit 2
fi
num_gpus=${NUM_GPUS:-2}
if [[ $num_gpus -ne 2 ]]; then
  echo "refusing world-size drift: expected 2 GPUs, got $num_gpus" >&2
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

control_config=examples/config/train_config_cfg90100_markov_control_s300.json
head_config=examples/config/train_config_cfg90100_markov_head_r32_s300.json
data_config=examples/config/data_config_emilia_full_blockparity.json
base_train_config=examples/config/train_config_cfg90100_band4_300k.json
final_checkpoint=${FINAL_CHECKPOINT:-/opt/gpfs/users/shuai/experiments/cfg90100-band4-pretrain-20260717/main300k_shuai-cfg90100-band4-300k-h8-v3/checkpoint-300000}
main_manifest=${MAIN_MANIFEST:-/opt/gpfs/users/shuai/experiments/cfg90100-band4-pretrain-20260717/main300k_shuai-cfg90100-band4-300k-h8-v3.manifest.txt}
main_source_commit=${EXPECTED_MAIN_SOURCE_COMMIT:-4a929df1174973b3b6b4ae66d7614443327d1054}

output_root=${OUTPUT_ROOT:-/opt/gpfs/users/shuai/experiments/dspark-markov300-ab-20260719}
if [[ -e $output_root ]]; then
  echo "refusing existing OUTPUT_ROOT: $output_root" >&2
  exit 1
fi
control_output="$output_root/control_${run_id}"
head_output="$output_root/head_r32_${run_id}"
control_replay_output="$output_root/control_replay_${run_id}"
log_dir="$output_root/logs"
report_dir="$output_root/reports"
manifest="$output_root/${run_id}.manifest.txt"
control_log="$log_dir/control_${run_id}.log"
head_log="$log_dir/head_r32_${run_id}.log"
control_replay_log="$log_dir/control_replay_${run_id}.log"
control_memory="$log_dir/control_${run_id}.memory.csv"
head_memory="$log_dir/head_r32_${run_id}.memory.csv"
control_replay_memory="$log_dir/control_replay_${run_id}.memory.csv"
contract_report="$report_dir/${run_id}.contract.log"
source_report="$report_dir/${run_id}.source.log"
vocab_report="$report_dir/${run_id}.vocab.log"
attach_report="$report_dir/${run_id}.attach.log"
gpu_inventory_report="$report_dir/${run_id}.gpu_inventory.csv"
training_report="$report_dir/${run_id}.training.json"
nll_report="$report_dir/${run_id}.nll.json"
generation_report="$report_dir/${run_id}.generation.json"
verdict_report="$report_dir/${run_id}.verdict.json"
receipt_report="$report_dir/${run_id}.receipt.json"
base_canary_root="$output_root/generation_frozen_base"
control_canary_root="$output_root/generation_control"
head_canary_root="$output_root/generation_head_r32"
mkdir -p "$log_dir" "$report_dir"

model_sha=$(sha256sum "$final_checkpoint/model.safetensors" | awk '{print $1}')
{
  echo "run_id=$run_id"
  echo "source_root=$root"
  echo "source_commit=$source_commit"
  echo "source_author=$(git show -s --format='%an <%ae>' HEAD)"
  echo "expected_main_source_commit=$main_source_commit"
  echo "control_config=$control_config"
  echo "control_config_sha256=$(sha256sum "$control_config" | awk '{print $1}')"
  echo "head_config=$head_config"
  echo "head_config_sha256=$(sha256sum "$head_config" | awk '{print $1}')"
  echo "base_train_config=$base_train_config"
  echo "base_train_config_sha256=$(sha256sum "$base_train_config" | awk '{print $1}')"
  echo "data_config=$data_config"
  echo "data_config_sha256=$(sha256sum "$data_config" | awk '{print $1}')"
  echo "main_manifest=$main_manifest"
  echo "main_manifest_sha256=$(sha256sum "$main_manifest" | awk '{print $1}')"
  echo "final_checkpoint=$final_checkpoint"
  echo "final_checkpoint_model_sha256=$model_sha"
  echo "num_gpus=$num_gpus"
  echo "control_output=$control_output"
  echo "head_output=$head_output"
  echo "control_replay_output=$control_replay_output"
  echo "base_canary_root=$base_canary_root"
  echo "control_canary_root=$control_canary_root"
  echo "head_canary_root=$head_canary_root"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$manifest"

phase=contract_preflight
python scripts/check_block_markov_ab_contract.py \
  --control-config "$control_config" \
  --head-config "$head_config" \
  --checkpoint "$final_checkpoint" \
  --manifest "$main_manifest" \
  --base-train-config "$base_train_config" \
  --data-config "$data_config" \
  --expected-main-source-commit "$main_source_commit" \
  2>&1 | tee "$contract_report"

phase=source_preflight
python - "$root" <<'PY' | tee "$source_report"
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

phase=vocab_preflight
python scripts/check_checkpoint_vocab.py \
  --train-config "$control_config" 2>&1 | tee "$vocab_report"
phase=attach_preflight
python scripts/check_block_markov_checkpoint_attach.py \
  --control-config "$control_config" \
  --checkpoint "$final_checkpoint" \
  --rank 32 2>&1 | tee "$attach_report"

phase=gpu_preflight
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
nvidia-smi --query-gpu=index,name,memory.total \
  --format=csv,noheader | tee "$gpu_inventory_report"

gpu_ids=$(seq -s, 0 $((num_gpus - 1)))
run_arm() {
  local arm=$1
  local config=$2
  local output=$3
  local log=$4
  local memory=$5

  phase="train_$arm"
  echo "BLOCK_MARKOV_ARM_START arm=$arm output=$output time=$(date -u +%FT%TZ)"
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used \
    --format=csv,noheader,nounits \
    --loop-ms=1000 > "$memory" &
  sampler_pid=$!
  set +e
  accelerate launch \
    --gpu_ids "$gpu_ids" \
    --num_processes "$num_gpus" \
    --num_machines 1 \
    -m omnivoice.cli.train \
    --train_config "$config" \
    --data_config "$data_config" \
    --output_dir "$output" \
    2>&1 | tee "$log"
  local pipeline_status=("${PIPESTATUS[@]}")
  local train_rc=${pipeline_status[0]}
  local tee_rc=${pipeline_status[1]}
  set -e
  if ! stop_sampler strict; then
    echo "GPU memory sampler shutdown failed for arm=$arm" >&2
    return 1
  fi
  if [[ $train_rc -ne 0 || $tee_rc -ne 0 ]]; then
    echo "BLOCK_MARKOV_ARM_FAILED arm=$arm train_rc=$train_rc tee_rc=$tee_rc" >&2
    return 1
  fi
  if [[ ! -s $memory ]]; then
    echo "GPU memory sampler produced no data for arm=$arm" >&2
    return 1
  fi
  echo "BLOCK_MARKOV_ARM_DONE arm=$arm time=$(date -u +%FT%TZ)"
}

run_arm control "$control_config" "$control_output" "$control_log" "$control_memory"
run_arm head_r32 "$head_config" "$head_output" "$head_log" "$head_memory"
run_arm control_replay "$control_config" "$control_replay_output" "$control_replay_log" "$control_replay_memory"

control_checkpoint="$control_output/checkpoint-300"
head_checkpoint="$head_output/checkpoint-300"
control_replay_checkpoint="$control_replay_output/checkpoint-300"
phase=training_report
python scripts/report_block_markov_training_ab.py \
  --control-log "$control_log" \
  --head-log "$head_log" \
  --control-replay-log "$control_replay_log" \
  --control-memory "$control_memory" \
  --head-memory "$head_memory" \
  --control-replay-memory "$control_replay_memory" \
  --control-checkpoint "$control_checkpoint" \
  --head-checkpoint "$head_checkpoint" \
  --control-replay-checkpoint "$control_replay_checkpoint" \
  --output "$training_report"

phase=nll_evaluation
python scripts/eval_block_markov_ab.py \
  --control-config "$control_config" \
  --head-config "$head_config" \
  --base-checkpoint "$final_checkpoint" \
  --control-checkpoint "$control_checkpoint" \
  --control-replay-checkpoint "$control_replay_checkpoint" \
  --head-checkpoint "$head_checkpoint" \
  --output "$nll_report" \
  --num-packs 32

run_canary() {
  local arm=$1
  local checkpoint=$2
  local result_root=$3
  phase="generation_canary_$arm"
  C="$root" \
  CK="$checkpoint" \
  CONFIG_SRC="$checkpoint/config.json" \
  MODE=shared_sweep \
  EXPECTED_COUNT=5 \
  GPU_IDS=0,1 \
  STEPS_PER_BLOCK=16 \
  RESULT_ROOT="$result_root" \
  RUN_ID="${arm}_${run_id}" \
    bash training_contract_cfg_guidance_probe.sh
  local result="$result_root/${arm}_${run_id}"
  for artifact in \
    "$result/SUMMARY.json" "$result/SUMMARY.tsv" "$result/SUMMARY.md" \
    "$result/PROVENANCE.txt" "$result/VERDICT.txt" \
    "$result/inputs/zh/manifest.json" "$result/inputs/en/manifest.json"; do
    if [[ ! -s $artifact ]]; then
      echo "generation canary artifact is missing or empty: $artifact" >&2
      return 1
    fi
  done
  if ! grep -q 'CFG_GUIDANCE_PROBE_DONE' "$result/VERDICT.txt"; then
    echo "generation canary did not reach completion marker: $result" >&2
    return 1
  fi
}

run_canary frozen_base "$final_checkpoint" "$base_canary_root"
run_canary control "$control_checkpoint" "$control_canary_root"
run_canary head_r32 "$head_checkpoint" "$head_canary_root"
base_canary_result="$base_canary_root/frozen_base_${run_id}"
control_canary_result="$control_canary_root/control_${run_id}"
head_canary_result="$head_canary_root/head_r32_${run_id}"
phase=generation_report
python scripts/eval_block_markov_generation_canary.py \
  --base-summary "$base_canary_result/SUMMARY.json" \
  --control-summary "$control_canary_result/SUMMARY.json" \
  --head-summary "$head_canary_result/SUMMARY.json" \
  --base-result "$base_canary_result" \
  --control-result "$control_canary_result" \
  --head-result "$head_canary_result" \
  --output "$generation_report"

phase=final_verdict
python - "$training_report" "$nll_report" "$generation_report" "$verdict_report" <<'PY'
import json
import sys
from pathlib import Path

training_path, nll_path, generation_path, output_path = map(Path, sys.argv[1:])
training = json.loads(training_path.read_text())
nll = json.loads(nll_path.read_text())
generation = json.loads(generation_path.read_text())
verdict = (
    "PROMOTE_TO_10K"
    if training["verdict"] == nll["verdict"] == generation["verdict"] == "PASS"
    else "KILL"
)
report = {
    "verdict": verdict,
    "scope": "300-step mechanism proof plus bilingual first5 disaster canary; no final quality claim",
    "training_verdict": training["verdict"],
    "nll_verdict": nll["verdict"],
    "generation_verdict": generation["verdict"],
    "training_report": str(training_path),
    "nll_report": str(nll_path),
    "generation_report": str(generation_path),
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print("BLOCK_MARKOV_VERDICT_COMPUTED " + json.dumps(report, sort_keys=True))
PY

scientific_verdict=$(python - "$verdict_report" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["verdict"])
PY
)
phase=receipt_finalize
python scripts/finalize_block_markov_receipt.py \
  --manifest "$manifest" \
  --output "$receipt_report" \
  --scientific-verdict "$scientific_verdict" \
  --artifact "control_config=$control_config" \
  --artifact "head_config=$head_config" \
  --artifact "base_train_config=$base_train_config" \
  --artifact "data_config=$data_config" \
  --artifact "main_manifest=$main_manifest" \
  --artifact "frozen_base_model=$final_checkpoint/model.safetensors" \
  --artifact "contract_report=$contract_report" \
  --artifact "source_report=$source_report" \
  --artifact "vocab_report=$vocab_report" \
  --artifact "attach_report=$attach_report" \
  --artifact "gpu_inventory_report=$gpu_inventory_report" \
  --artifact "control_log=$control_log" \
  --artifact "head_log=$head_log" \
  --artifact "control_replay_log=$control_replay_log" \
  --artifact "control_memory=$control_memory" \
  --artifact "head_memory=$head_memory" \
  --artifact "control_replay_memory=$control_replay_memory" \
  --artifact "control_model=$control_checkpoint/model.safetensors" \
  --artifact "head_model=$head_checkpoint/model.safetensors" \
  --artifact "control_replay_model=$control_replay_checkpoint/model.safetensors" \
  --artifact "training_report=$training_report" \
  --artifact "nll_report=$nll_report" \
  --artifact "generation_report=$generation_report" \
  --artifact "verdict_report=$verdict_report" \
  --artifact "base_summary_json=$base_canary_result/SUMMARY.json" \
  --artifact "base_summary_tsv=$base_canary_result/SUMMARY.tsv" \
  --artifact "base_summary_md=$base_canary_result/SUMMARY.md" \
  --artifact "base_provenance=$base_canary_result/PROVENANCE.txt" \
  --artifact "base_verdict=$base_canary_result/VERDICT.txt" \
  --artifact "base_input_zh_manifest=$base_canary_result/inputs/zh/manifest.json" \
  --artifact "base_input_en_manifest=$base_canary_result/inputs/en/manifest.json" \
  --artifact "control_summary_json=$control_canary_result/SUMMARY.json" \
  --artifact "control_summary_tsv=$control_canary_result/SUMMARY.tsv" \
  --artifact "control_summary_md=$control_canary_result/SUMMARY.md" \
  --artifact "control_provenance=$control_canary_result/PROVENANCE.txt" \
  --artifact "control_verdict=$control_canary_result/VERDICT.txt" \
  --artifact "control_input_zh_manifest=$control_canary_result/inputs/zh/manifest.json" \
  --artifact "control_input_en_manifest=$control_canary_result/inputs/en/manifest.json" \
  --artifact "head_summary_json=$head_canary_result/SUMMARY.json" \
  --artifact "head_summary_tsv=$head_canary_result/SUMMARY.tsv" \
  --artifact "head_summary_md=$head_canary_result/SUMMARY.md" \
  --artifact "head_provenance=$head_canary_result/PROVENANCE.txt" \
  --artifact "head_verdict=$head_canary_result/VERDICT.txt" \
  --artifact "head_input_zh_manifest=$head_canary_result/inputs/zh/manifest.json" \
  --artifact "head_input_en_manifest=$head_canary_result/inputs/en/manifest.json"
if [[ $scientific_verdict == PROMOTE_TO_10K ]]; then
  echo "BLOCK_MARKOV_PROMOTE_TO_10K verdict_report=$verdict_report receipt=$receipt_report"
else
  echo "BLOCK_MARKOV_SCIENTIFIC_KILL verdict_report=$verdict_report receipt=$receipt_report"
fi
finished=1
echo "BLOCK_MARKOV_EXIT run_id=$run_id verdict=$scientific_verdict rc=0 time=$(date -u +%FT%TZ)"
