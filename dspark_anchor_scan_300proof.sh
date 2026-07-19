#!/usr/bin/env bash
# Self-terminating seed-0 mechanism proof. This script never submits follow-ups.
# OMS must invoke it as flat argv:
# /usr/bin/env KEY=VALUE ... /bin/bash /absolute/dspark_anchor_scan_300proof.sh
set -euo pipefail

phase=preflight
manifest=""
sampler_pid=""
finished=0

stop_sampler() {
  local strict=${1:-1}
  local pid=${sampler_pid:-}
  [[ -n $pid ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
  set +e
  wait "$pid"
  local rc=$?
  set -e
  sampler_pid=""
  if [[ $strict -eq 1 && $rc -ne 0 && $rc -ne 137 && $rc -ne 143 ]]; then
    echo "GPU sampler failed rc=$rc" >&2
    return 1
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  [[ -z $sampler_pid ]] || stop_sampler 0 || true
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
  echo "usage: dspark_anchor_scan_300proof.sh" >&2
  exit 2
fi
run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ $run_id =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid RUN_ID" >&2; exit 2; }
[[ $run_id == shuai-* ]] || { echo "RUN_ID must use the shuai- prefix" >&2; exit 2; }
num_gpus=${NUM_GPUS:-2}
[[ $num_gpus -eq 2 ]] || { echo "expected exactly 2 GPUs" >&2; exit 2; }

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
expected_root=${EXPECTED_SOURCE_ROOT:?EXPECTED_SOURCE_ROOT is required}
[[ $(realpath "$root") == $(realpath "$expected_root") ]] || {
  echo "unexpected source root: $root" >&2; exit 1;
}
cd "$root"
[[ -z $(git status --porcelain) ]] || { git status --short >&2; exit 1; }
source_commit=$(git rev-parse HEAD)
expected_commit=${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required}
[[ $source_commit == "$expected_commit" ]] || {
  echo "source commit drift expected=$expected_commit actual=$source_commit" >&2; exit 1;
}

runtime_venv=${RUNTIME_VENV:-/opt/gpfs/users/yinfeng/work/OmniVoice/.venv}
source "$runtime_venv/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$root:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

causal_config=examples/config/train_config_cfg90100_anchor_causal_s300.json
stateless_config=examples/config/train_config_cfg90100_anchor_stateless_s300.json
data_config=examples/config/data_config_emilia_full_blockparity.json
base_train_config=examples/config/train_config_cfg90100_band4_300k.json
base_checkpoint=${FINAL_CHECKPOINT:-/opt/gpfs/users/shuai/experiments/cfg90100-band4-pretrain-20260717/main300k_shuai-cfg90100-band4-300k-h8-v3/checkpoint-300000}
main_manifest=${MAIN_MANIFEST:-/opt/gpfs/users/shuai/experiments/cfg90100-band4-pretrain-20260717/main300k_shuai-cfg90100-band4-300k-h8-v3.manifest.txt}
main_source_commit=${EXPECTED_MAIN_SOURCE_COMMIT:-4a929df1174973b3b6b4ae66d7614443327d1054}
output_root=$(realpath -m -- "${OUTPUT_ROOT:-/opt/gpfs/users/shuai/experiments/dspark-anchor-seed0-300proof-20260719}")
case "$output_root" in
  /opt/gpfs/users/shuai/experiments/dspark-anchor-*) ;;
  *) echo "OUTPUT_ROOT must be a new Shuai dspark-anchor experiment: $output_root" >&2; exit 2 ;;
esac
mkdir "$output_root" || {
  echo "refusing non-new OUTPUT_ROOT: $output_root" >&2
  exit 1
}

causal_output="$output_root/causal_${run_id}"
stateless_output="$output_root/stateless_${run_id}"
replay_output="$output_root/causal_replay_${run_id}"
log_dir="$output_root/logs"
report_dir="$output_root/reports"
mkdir "$log_dir" "$report_dir"
manifest="$output_root/${run_id}.manifest.txt"
contract_report="$report_dir/${run_id}.contract.log"
contract_post_report="$report_dir/${run_id}.contract_post.log"
source_report="$report_dir/${run_id}.source.log"
vocab_report="$report_dir/${run_id}.vocab.log"
attach_report="$report_dir/${run_id}.attach.log"
environment_report="$report_dir/${run_id}.environment.json"
gpu_report="$report_dir/${run_id}.gpu.csv"
training_report="$report_dir/${run_id}.training.json"
nll_report="$report_dir/${run_id}.nll.json"
verdict_report="$report_dir/${run_id}.verdict.json"
receipt_report="$report_dir/${run_id}.receipt.json"

declare -A logs memories
logs[causal]="$log_dir/causal_${run_id}.log"
logs[stateless]="$log_dir/stateless_${run_id}.log"
logs[causal_replay]="$log_dir/causal_replay_${run_id}.log"
memories[causal]="$log_dir/causal_${run_id}.memory.csv"
memories[stateless]="$log_dir/stateless_${run_id}.memory.csv"
memories[causal_replay]="$log_dir/causal_replay_${run_id}.memory.csv"

{
  echo "run_id=$run_id"
  echo "scope=seed0_300step_only"
  echo "seed_index=0"
  echo "train_seed=42"
  echo "eval_seed=20260719"
  echo "english_only_mechanism_probe=1"
  echo "replay_contract=bitwise_signal_with_numeric_noise_floor"
  echo "source_root=$root"
  echo "source_commit=$source_commit"
  echo "source_author=$(git show -s --format='%an <%ae>' HEAD)"
  echo "base_checkpoint=$base_checkpoint"
  echo "base_model_sha256=$(sha256sum "$base_checkpoint/model.safetensors" | awk '{print $1}')"
  echo "causal_config=$causal_config"
  echo "causal_config_sha256=$(sha256sum "$causal_config" | awk '{print $1}')"
  echo "stateless_config=$stateless_config"
  echo "stateless_config_sha256=$(sha256sum "$stateless_config" | awk '{print $1}')"
  echo "causal_output=$causal_output"
  echo "stateless_output=$stateless_output"
  echo "causal_replay_output=$replay_output"
  echo "automatic_followup_submitted=0"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$manifest"

phase=contract
python scripts/check_block_anchor_contract.py \
  --causal-config "$causal_config" \
  --stateless-config "$stateless_config" \
  --checkpoint "$base_checkpoint" \
  --manifest "$main_manifest" \
  --base-train-config "$base_train_config" \
  --data-config "$data_config" \
  --expected-main-source-commit "$main_source_commit" 2>&1 | tee "$contract_report"
data_manifest_inventory_sha256=$(python - "$contract_report" <<'PY'
import json, sys
prefix = "BLOCK_ANCHOR_CONTRACT_OK "
lines = [line for line in open(sys.argv[1]) if line.startswith(prefix)]
if len(lines) != 1:
    raise SystemExit(f"expected exactly one contract marker, got {len(lines)}")
print(json.loads(lines[0][len(prefix):])["data_manifest_inventory_sha256"])
PY
)
echo "data_manifest_inventory_sha256=$data_manifest_inventory_sha256" >> "$manifest"

phase=source
python - "$root" <<'PY' | tee "$source_report"
import pathlib, sys, omnivoice
expected = pathlib.Path(sys.argv[1]).resolve()
actual = pathlib.Path(omnivoice.__file__).resolve()
print(f"python={sys.executable}")
print(f"omnivoice={actual}")
if expected not in actual.parents:
    raise SystemExit(f"refusing non-Shuai source: {actual}")
PY

phase=environment
python - "$source_commit" "$environment_report" <<'PY'
import hashlib, json, platform, subprocess, sys
from pathlib import Path

import accelerate
import liger_kernel
import torch
import transformers

commit, output = sys.argv[1:]
gpu_rows = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip().splitlines()
driver_rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    text=True,
).strip().splitlines()
liger_root = Path(liger_kernel.__file__).resolve().parent
liger_files = sorted(
    path
    for path in liger_root.rglob("*")
    if path.is_file()
    and "__pycache__" not in path.parts
    and path.suffix not in {".pyc", ".pyo"}
)
liger_digest = hashlib.sha256()
for path in liger_files:
    relative = path.relative_to(liger_root).as_posix()
    liger_digest.update(relative.encode())
    liger_digest.update(b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            liger_digest.update(chunk)
payload = {
    "source_commit": commit,
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "transformers_version": transformers.__version__,
    "accelerate_version": accelerate.__version__,
    "liger_kernel_source_root": str(liger_root),
    "liger_kernel_file_count": len(liger_files),
    "liger_kernel_tree_sha256": liger_digest.hexdigest(),
    "nvidia_driver_version": sorted(set(driver_rows)),
    "gpu_inventory": gpu_rows,
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("BLOCK_ANCHOR_ENVIRONMENT_OK " + json.dumps(payload, sort_keys=True))
PY

phase=vocab
python scripts/check_checkpoint_vocab.py --train-config "$causal_config" 2>&1 | tee "$vocab_report"
phase=attach
python scripts/check_block_anchor_checkpoint_attach.py \
  --causal-config "$causal_config" --checkpoint "$base_checkpoint" \
  2>&1 | tee "$attach_report"
phase=gpu
gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $(printf '%s\n' "$gpu_names" | sed '/^$/d' | wc -l) -eq 2 ]] || exit 1
printf '%s\n' "$gpu_names" | grep -vi H100 >/dev/null && exit 1 || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee "$gpu_report"

gpu_ids=0,1
run_arm() {
  local arm=$1 config=$2 output=$3
  local log=${logs[$arm]} memory=${memories[$arm]}
  phase="train_$arm"
  echo "BLOCK_ANCHOR_ARM_START arm=$arm time=$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=timestamp,index,memory.used \
    --format=csv,noheader,nounits --loop-ms=1000 > "$memory" &
  sampler_pid=$!
  set +e
  accelerate launch --gpu_ids "$gpu_ids" --num_processes 2 --num_machines 1 \
    -m omnivoice.cli.train --train_config "$config" --data_config "$data_config" \
    --output_dir "$output" 2>&1 | tee "$log"
  local status=("${PIPESTATUS[@]}")
  set -e
  stop_sampler 1
  [[ ${status[0]} -eq 0 && ${status[1]} -eq 0 ]] || {
    echo "BLOCK_ANCHOR_ARM_FAILED arm=$arm train_rc=${status[0]} tee_rc=${status[1]}" >&2
    return 1
  }
  [[ -s $memory ]] || { echo "empty memory trace for $arm" >&2; return 1; }
  echo "BLOCK_ANCHOR_ARM_DONE arm=$arm time=$(date -u +%FT%TZ)"
}

run_arm causal "$causal_config" "$causal_output"
run_arm stateless "$stateless_config" "$stateless_output"
run_arm causal_replay "$causal_config" "$replay_output"

causal_checkpoint="$causal_output/checkpoint-300"
stateless_checkpoint="$stateless_output/checkpoint-300"
replay_checkpoint="$replay_output/checkpoint-300"
phase=training_report
python scripts/report_block_anchor_training.py \
  --causal-log "${logs[causal]}" --stateless-log "${logs[stateless]}" \
  --causal-replay-log "${logs[causal_replay]}" \
  --causal-memory "${memories[causal]}" --stateless-memory "${memories[stateless]}" \
  --causal-replay-memory "${memories[causal_replay]}" \
  --base-checkpoint "$base_checkpoint" --causal-checkpoint "$causal_checkpoint" \
  --stateless-checkpoint "$stateless_checkpoint" \
  --causal-replay-checkpoint "$replay_checkpoint" --output "$training_report"

phase=nll
python scripts/eval_block_anchor_nll.py \
  --causal-config "$causal_config" --stateless-config "$stateless_config" \
  --base-checkpoint "$base_checkpoint" --causal-checkpoint "$causal_checkpoint" \
  --stateless-checkpoint "$stateless_checkpoint" --num-packs 32 \
  --benchmark-packs 4 --benchmark-warmup 2 --benchmark-repeats 5 \
  --output "$nll_report"

phase=contract_post
python scripts/check_block_anchor_contract.py \
  --causal-config "$causal_config" \
  --stateless-config "$stateless_config" \
  --checkpoint "$base_checkpoint" \
  --manifest "$main_manifest" \
  --base-train-config "$base_train_config" \
  --data-config "$data_config" \
  --expected-main-source-commit "$main_source_commit" 2>&1 | tee "$contract_post_report"
post_data_manifest_inventory_sha256=$(python - "$contract_post_report" <<'PY'
import json, sys
prefix = "BLOCK_ANCHOR_CONTRACT_OK "
lines = [line for line in open(sys.argv[1]) if line.startswith(prefix)]
if len(lines) != 1:
    raise SystemExit(f"expected exactly one post-contract marker, got {len(lines)}")
print(json.loads(lines[0][len(prefix):])["data_manifest_inventory_sha256"])
PY
)
[[ $post_data_manifest_inventory_sha256 == "$data_manifest_inventory_sha256" ]] || {
  echo "data manifest content drifted during proof" >&2
  exit 1
}
echo "post_data_manifest_inventory_sha256=$post_data_manifest_inventory_sha256" >> "$manifest"

phase=verdict
python - "$training_report" "$nll_report" "$verdict_report" <<'PY'
import json, sys
from pathlib import Path
from scripts.finalize_block_anchor_receipt import resolve_final_verdict
training_path, nll_path, output = map(Path, sys.argv[1:])
training = json.loads(training_path.read_text())
nll = json.loads(nll_path.read_text())
verdict = resolve_final_verdict(training["verdict"], nll["verdict"])
allowed = {"PROMOTE_TO_3SEED", "SCIENTIFIC_KILL", "INCONCLUSIVE_1K", "INVALID_IMPLEMENTATION", "ENGINEERING_BLOCK"}
if verdict not in allowed:
    raise SystemExit(f"invalid final verdict: {verdict}")
report = {
    "verdict": verdict,
    "training_verdict": training["verdict"],
    "nll_verdict": nll["verdict"],
    "generation_status": "NEEDS_GENERATION",
    "automatic_followup_submitted": False,
    "scope": "seed-0 300-step mechanism proof only",
    "seed_index": 0,
    "train_seed": 42,
    "eval_seed": 20260719,
    "english_only_mechanism_probe": True,
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print("BLOCK_ANCHOR_FINAL_VERDICT " + json.dumps(report, sort_keys=True))
PY

scientific_verdict=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$verdict_report")
phase=receipt
python scripts/finalize_block_anchor_receipt.py \
  --manifest "$manifest" --output "$receipt_report" \
  --scientific-verdict "$scientific_verdict" \
  --verdict-report "$verdict_report" \
  --training-report "$training_report" \
  --nll-report "$nll_report" \
  --environment-report "$environment_report" \
  --seed-index 0 --train-seed 42 --eval-seed 20260719 \
  --english-only-mechanism-probe \
  --checkpoint "base=$base_checkpoint" \
  --checkpoint "causal=$causal_checkpoint" \
  --checkpoint "stateless=$stateless_checkpoint" \
  --checkpoint "causal_replay=$replay_checkpoint" \
  --artifact "causal_config=$causal_config" \
  --artifact "stateless_config=$stateless_config" \
  --artifact "base_train_config=$base_train_config" \
  --artifact "data_config=$data_config" \
  --artifact "main_manifest=$main_manifest" \
  --artifact "base_model=$base_checkpoint/model.safetensors" \
  --artifact "contract_report=$contract_report" \
  --artifact "contract_post_report=$contract_post_report" \
  --artifact "source_report=$source_report" \
  --artifact "vocab_report=$vocab_report" \
  --artifact "attach_report=$attach_report" \
  --artifact "environment_report=$environment_report" \
  --artifact "gpu_report=$gpu_report" \
  --artifact "causal_log=${logs[causal]}" \
  --artifact "stateless_log=${logs[stateless]}" \
  --artifact "causal_replay_log=${logs[causal_replay]}" \
  --artifact "causal_memory=${memories[causal]}" \
  --artifact "stateless_memory=${memories[stateless]}" \
  --artifact "causal_replay_memory=${memories[causal_replay]}" \
  --artifact "causal_model=$causal_checkpoint/model.safetensors" \
  --artifact "stateless_model=$stateless_checkpoint/model.safetensors" \
  --artifact "causal_replay_model=$replay_checkpoint/model.safetensors" \
  --artifact "training_report=$training_report" \
  --artifact "nll_report=$nll_report" \
  --artifact "verdict_report=$verdict_report"

echo "BLOCK_ANCHOR_RESULT verdict=$scientific_verdict generation=NEEDS_GENERATION receipt=$receipt_report"
echo "BLOCK_ANCHOR_NO_AUTOMATIC_FOLLOWUP"
{
  echo "finished_utc=$(date -u +%FT%TZ)"
  echo "proof_complete=1"
  echo "train_rc=0"
  echo "tee_rc=0"
  echo "infrastructure_rc=0"
  echo "rc=0"
} >> "$manifest"
finished=1
echo "BLOCK_ANCHOR_EXIT run_id=$run_id rc=0 time=$(date -u +%FT%TZ)"
