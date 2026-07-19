#!/usr/bin/env bash
# Eval-only recovery for a seed-0 anchor proof whose three training arms
# completed but whose original process failed before producing an NLL verdict.
# This script never trains, mutates the parent proof, or submits follow-ups.
set -euo pipefail

phase=preflight
manifest=""
finished=0
preflight_snapshot=""
source_preflight_snapshot=""

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  [[ -z $preflight_snapshot ]] || rm -f -- "$preflight_snapshot"
  [[ -z $source_preflight_snapshot ]] || rm -f -- "$source_preflight_snapshot"
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
  echo "usage: dspark_anchor_scan_recover_eval.sh" >&2
  exit 2
fi

run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ $run_id =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid RUN_ID" >&2; exit 2; }
[[ $run_id == shuai-* ]] || { echo "RUN_ID must use the shuai- prefix" >&2; exit 2; }
num_gpus=${NUM_GPUS:-1}
[[ $num_gpus -eq 1 ]] || { echo "eval recovery requires exactly 1 GPU" >&2; exit 2; }

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
expected_root=${EXPECTED_SOURCE_ROOT:?EXPECTED_SOURCE_ROOT is required}
[[ $(realpath "$root") == $(realpath "$expected_root") ]] || {
  echo "unexpected source root: $root" >&2
  exit 1
}
cd "$root"
[[ -z $(git status --porcelain) ]] || { git status --short >&2; exit 1; }
source_commit=$(git rev-parse HEAD)
expected_commit=${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required}
[[ $source_commit == "$expected_commit" ]] || {
  echo "source commit drift expected=$expected_commit actual=$source_commit" >&2
  exit 1
}

parent_run_id=${PARENT_RUN_ID:?PARENT_RUN_ID is required}
[[ $parent_run_id =~ ^shuai-[A-Za-z0-9._-]+$ ]] || {
  echo "invalid PARENT_RUN_ID" >&2
  exit 2
}
[[ $parent_run_id != "$run_id" ]] || {
  echo "recovery RUN_ID must differ from PARENT_RUN_ID" >&2
  exit 2
}
parent_root=$(realpath -e -- "${PARENT_ROOT:?PARENT_ROOT is required}")
case "$parent_root" in
  /opt/gpfs/users/shuai/experiments/dspark-anchor-*) ;;
  *) echo "PARENT_ROOT must be a Shuai dspark-anchor result: $parent_root" >&2; exit 2 ;;
esac
expected_parent_source_commit=${EXPECTED_PARENT_SOURCE_COMMIT:?EXPECTED_PARENT_SOURCE_COMMIT is required}
expected_parent_manifest_sha256=${EXPECTED_PARENT_MANIFEST_SHA256:?EXPECTED_PARENT_MANIFEST_SHA256 is required}
expected_parent_training_sha256=${EXPECTED_PARENT_TRAINING_REPORT_SHA256:?EXPECTED_PARENT_TRAINING_REPORT_SHA256 is required}
expected_parent_causal_inventory_sha256=${EXPECTED_PARENT_CAUSAL_INVENTORY_SHA256:?EXPECTED_PARENT_CAUSAL_INVENTORY_SHA256 is required}
expected_parent_stateless_inventory_sha256=${EXPECTED_PARENT_STATELESS_INVENTORY_SHA256:?EXPECTED_PARENT_STATELESS_INVENTORY_SHA256 is required}
expected_parent_replay_inventory_sha256=${EXPECTED_PARENT_REPLAY_INVENTORY_SHA256:?EXPECTED_PARENT_REPLAY_INVENTORY_SHA256 is required}
parent_failure_log=$(realpath -e -- "${PARENT_FAILURE_LOG:?PARENT_FAILURE_LOG is required}")
expected_parent_failure_log_sha256=${EXPECTED_PARENT_FAILURE_LOG_SHA256:?EXPECTED_PARENT_FAILURE_LOG_SHA256 is required}
for expected_sha in \
  "$expected_parent_manifest_sha256" \
  "$expected_parent_training_sha256" \
  "$expected_parent_failure_log_sha256" \
  "$expected_parent_causal_inventory_sha256" \
  "$expected_parent_stateless_inventory_sha256" \
  "$expected_parent_replay_inventory_sha256"; do
  [[ $expected_sha =~ ^[0-9a-f]{64}$ ]] || {
    echo "expected SHA-256 values must be lowercase 64-hex strings" >&2
    exit 2
  }
done
[[ $source_commit != "$expected_parent_source_commit" ]] || {
  echo "recovery source commit must differ from parent source commit" >&2
  exit 1
}
mapfile -t changed_paths < <(git diff --name-only "$expected_parent_source_commit" "$source_commit" | sort)
expected_changed_paths=(
  dspark_anchor_scan_recover_eval.sh
  scripts/check_block_anchor_checkpoint_attach.py
  scripts/eval_block_anchor_nll.py
  scripts/finalize_block_anchor_receipt.py
  tests/test_block_anchor_campaign.py
  tests/test_block_anchor_recovery.py
)
mapfile -t expected_changed_paths < <(printf '%s\n' "${expected_changed_paths[@]}" | sort)
[[ ${changed_paths[*]} == "${expected_changed_paths[*]}" ]] || {
  printf 'recovery source diff escaped allowlist\nactual=%s\nexpected=%s\n' \
    "${changed_paths[*]}" "${expected_changed_paths[*]}" >&2
  exit 1
}

[[ -f $parent_failure_log && -s $parent_failure_log ]] || {
  echo "parent failure log is missing or empty: $parent_failure_log" >&2
  exit 1
}
actual_parent_failure_log_sha256=$(sha256sum "$parent_failure_log" | awk '{print $1}')
[[ $actual_parent_failure_log_sha256 == "$expected_parent_failure_log_sha256" ]] || {
  echo "parent failure log hash drift" >&2
  exit 1
}
for fragment in \
  _benchmark_head_on_off \
  _forward_device \
  'ValueError: Expected query, key, and value to have the same dtype' \
  'query.dtype: torch.float32' \
  'key.dtype: torch.float32' \
  'value.dtype: torch.bfloat16'; do
  grep -Fq "$fragment" "$parent_failure_log" || {
    echo "parent failure log lacks required signature: $fragment" >&2
    exit 1
  }
done

causal_config=examples/config/train_config_cfg90100_anchor_causal_s300.json
stateless_config=examples/config/train_config_cfg90100_anchor_stateless_s300.json
data_config=examples/config/data_config_emilia_full_blockparity.json
base_train_config=examples/config/train_config_cfg90100_band4_300k.json
base_checkpoint=${FINAL_CHECKPOINT:-/opt/gpfs/users/shuai/experiments/cfg90100-band4-pretrain-20260717/main300k_shuai-cfg90100-band4-300k-h8-v3/checkpoint-300000}
main_manifest=${MAIN_MANIFEST:-/opt/gpfs/users/shuai/experiments/cfg90100-band4-pretrain-20260717/main300k_shuai-cfg90100-band4-300k-h8-v3.manifest.txt}
main_source_commit=${EXPECTED_MAIN_SOURCE_COMMIT:-4a929df1174973b3b6b4ae66d7614443327d1054}

runtime_venv=${RUNTIME_VENV:-/opt/gpfs/users/yinfeng/work/OmniVoice/.venv}
source "$runtime_venv/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$root:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

parent_manifest="$parent_root/${parent_run_id}.manifest.txt"
parent_training_report="$parent_root/reports/${parent_run_id}.training.json"
parent_causal_log="$parent_root/logs/causal_${parent_run_id}.log"
parent_stateless_log="$parent_root/logs/stateless_${parent_run_id}.log"
parent_replay_log="$parent_root/logs/causal_replay_${parent_run_id}.log"
parent_causal_memory="$parent_root/logs/causal_${parent_run_id}.memory.csv"
parent_stateless_memory="$parent_root/logs/stateless_${parent_run_id}.memory.csv"
parent_replay_memory="$parent_root/logs/causal_replay_${parent_run_id}.memory.csv"
parent_causal_checkpoint="$parent_root/causal_${parent_run_id}/checkpoint-300"
parent_stateless_checkpoint="$parent_root/stateless_${parent_run_id}/checkpoint-300"
parent_replay_checkpoint="$parent_root/causal_replay_${parent_run_id}/checkpoint-300"

validate_parent() {
  local output=$1
  python - \
    "$parent_root" "$parent_run_id" "$expected_parent_source_commit" \
    "$expected_parent_manifest_sha256" "$expected_parent_training_sha256" \
    "$expected_parent_causal_inventory_sha256" \
    "$expected_parent_stateless_inventory_sha256" \
    "$expected_parent_replay_inventory_sha256" \
    "$base_checkpoint" "$causal_config" "$stateless_config" \
    "$parent_failure_log" "$expected_parent_failure_log_sha256" "$output" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    parent_root_raw,
    run_id,
    expected_source_commit,
    expected_manifest_sha,
    expected_training_sha,
    expected_causal_inventory,
    expected_stateless_inventory,
    expected_replay_inventory,
    base_checkpoint_raw,
    causal_config_raw,
    stateless_config_raw,
    parent_failure_log_raw,
    expected_parent_failure_log_sha,
    output_raw,
) = sys.argv[1:]
parent_root = Path(parent_root_raw).resolve(strict=True)
base_checkpoint = Path(base_checkpoint_raw).resolve(strict=True)
causal_config = Path(causal_config_raw).resolve(strict=True)
stateless_config = Path(stateless_config_raw).resolve(strict=True)
parent_failure_log = Path(parent_failure_log_raw).resolve(strict=True)
output = Path(output_raw)


def sha256(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"parent artifact missing or empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_inventory(path: Path) -> dict:
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise SystemExit(f"parent checkpoint is not a directory: {path}")
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise SystemExit(f"parent checkpoint inventory forbids symlinks: {item}")
        if not item.is_file():
            continue
        if item.stat().st_size == 0:
            raise SystemExit(f"parent checkpoint contains empty file: {item}")
        files.append(
            {
                "relative_path": item.relative_to(path).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": sha256(item),
            }
        )
    required = {
        "config.json",
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "train_config.json",
    }
    names = {item["relative_path"] for item in files}
    missing = sorted(required - names)
    random_states = sorted(name for name in names if name.startswith("random_states_") and name.endswith(".pkl"))
    if missing or random_states != ["random_states_0.pkl", "random_states_1.pkl"]:
        raise SystemExit(
            f"incomplete parent checkpoint {path}: missing={missing} random_states={random_states}"
        )
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    return {
        "path": str(path),
        "file_count": len(files),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


manifest = parent_root / f"{run_id}.manifest.txt"
training = parent_root / "reports" / f"{run_id}.training.json"
causal_log = parent_root / "logs" / f"causal_{run_id}.log"
stateless_log = parent_root / "logs" / f"stateless_{run_id}.log"
replay_log = parent_root / "logs" / f"causal_replay_{run_id}.log"
causal_memory = parent_root / "logs" / f"causal_{run_id}.memory.csv"
stateless_memory = parent_root / "logs" / f"stateless_{run_id}.memory.csv"
replay_memory = parent_root / "logs" / f"causal_replay_{run_id}.memory.csv"
causal_checkpoint = parent_root / f"causal_{run_id}" / "checkpoint-300"
stateless_checkpoint = parent_root / f"stateless_{run_id}" / "checkpoint-300"
replay_checkpoint = parent_root / f"causal_replay_{run_id}" / "checkpoint-300"

manifest_sha = sha256(manifest)
training_sha = sha256(training)
parent_failure_log_sha = sha256(parent_failure_log)
if manifest_sha != expected_manifest_sha:
    raise SystemExit(
        f"parent manifest hash drift: expected={expected_manifest_sha} actual={manifest_sha}"
    )
if training_sha != expected_training_sha:
    raise SystemExit(
        f"parent training report hash drift: expected={expected_training_sha} actual={training_sha}"
    )
if parent_failure_log_sha != expected_parent_failure_log_sha:
    raise SystemExit(
        "parent failure log hash drift: "
        f"expected={expected_parent_failure_log_sha} actual={parent_failure_log_sha}"
    )

values: dict[str, list[str]] = {}
for line in manifest.read_text().splitlines():
    if "=" not in line:
        raise SystemExit(f"malformed parent manifest line: {line!r}")
    key, value = line.split("=", 1)
    values.setdefault(key, []).append(value)


def exactly(key: str, expected: str) -> None:
    actual = values.get(key)
    if actual != [expected]:
        raise SystemExit(
            f"parent manifest mismatch for {key}: expected={[expected]!r} actual={actual!r}"
        )


exactly("run_id", run_id)
exactly("scope", "seed0_300step_only")
exactly("seed_index", "0")
exactly("train_seed", "42")
exactly("eval_seed", "20260719")
exactly("english_only_mechanism_probe", "1")
exactly("source_commit", expected_source_commit)
exactly("base_checkpoint", str(base_checkpoint))
exactly("base_model_sha256", sha256(base_checkpoint / "model.safetensors"))
exactly("causal_config", "examples/config/train_config_cfg90100_anchor_causal_s300.json")
exactly("stateless_config", "examples/config/train_config_cfg90100_anchor_stateless_s300.json")
exactly("causal_config_sha256", sha256(causal_config))
exactly("stateless_config_sha256", sha256(stateless_config))
exactly("causal_output", str(parent_root / f"causal_{run_id}"))
exactly("stateless_output", str(parent_root / f"stateless_{run_id}"))
exactly("causal_replay_output", str(parent_root / f"causal_replay_{run_id}"))
exactly("automatic_followup_submitted", "0")
exactly("failed_phase", "nll")
exactly("infrastructure_rc", "1")
for required_nonempty in (
    "data_manifest_inventory_sha256",
    "finished_utc",
    "source_author",
    "source_root",
    "started_utc",
):
    actual = values.get(required_nonempty)
    if actual is None or len(actual) != 1 or not actual[0]:
        raise SystemExit(f"parent manifest field is missing/duplicated/empty: {required_nonempty}")
for forbidden in (
    "artifact_receipt",
    "artifact_receipt_sha256",
    "proof_complete",
    "scientific_verdict",
    "train_rc",
    "tee_rc",
    "rc",
):
    if forbidden in values:
        raise SystemExit(f"parent proof was already finalized: unexpected {forbidden}")

training_payload = json.loads(training.read_text())
if training_payload.get("verdict") != "PASS":
    raise SystemExit(
        f"parent training report is not reusable: verdict={training_payload.get('verdict')!r}"
    )
if not training_payload.get("frozen_backbone_hash_exact"):
    raise SystemExit("parent training report lacks exact frozen-backbone proof")
if not training_payload.get("causal_replay_bitwise_exact"):
    raise SystemExit("parent training report lacks bitwise replay proof")

inventories = {
    "causal": checkpoint_inventory(causal_checkpoint),
    "stateless": checkpoint_inventory(stateless_checkpoint),
    "causal_replay": checkpoint_inventory(replay_checkpoint),
}
expected_inventories = {
    "causal": expected_causal_inventory,
    "stateless": expected_stateless_inventory,
    "causal_replay": expected_replay_inventory,
}
for name, inventory in inventories.items():
    expected = expected_inventories[name]
    actual = inventory["inventory_sha256"]
    if actual != expected:
        raise SystemExit(
            f"parent {name} checkpoint inventory drift: expected={expected} actual={actual}"
        )

artifact_paths = {
    "causal_log": causal_log,
    "stateless_log": stateless_log,
    "causal_replay_log": replay_log,
    "causal_memory": causal_memory,
    "stateless_memory": stateless_memory,
    "causal_replay_memory": replay_memory,
    "failure_log": parent_failure_log,
}
artifacts = {
    name: {"path": str(path.resolve(strict=True)), "sha256": sha256(path)}
    for name, path in artifact_paths.items()
}
payload = {
    "schema": "block_anchor_eval_recovery_parent_lock_v1",
    "parent_root": str(parent_root),
    "parent_run_id": run_id,
    "parent_source_root": values["source_root"][0],
    "parent_source_commit": expected_source_commit,
    "parent_manifest": str(manifest),
    "parent_manifest_sha256": manifest_sha,
    "parent_training_report": str(training),
    "parent_training_report_sha256": training_sha,
    "parent_failure_log": str(parent_failure_log),
    "parent_failure_log_sha256": parent_failure_log_sha,
    "parent_failed_phase": "nll",
    "parent_infrastructure_rc": 1,
    "parent_automatic_followup_submitted": False,
    "training_reused": True,
    "training_reexecuted": False,
    "artifacts": dict(sorted(artifacts.items())),
    "checkpoint_inventories": dict(sorted(inventories.items())),
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("BLOCK_ANCHOR_RECOVERY_PARENT_OK " + json.dumps(payload, sort_keys=True))
PY
}

validate_source() {
  local output=$1
  python - "$root" "$source_commit" "$expected_parent_source_commit" "$output" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root_raw, expected_commit, parent_commit, output_raw = sys.argv[1:]
root = Path(root_raw).resolve(strict=True)
output = Path(output_raw)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


actual_commit = git("rev-parse", "HEAD")
if actual_commit != expected_commit:
    raise SystemExit(
        f"recovery source HEAD drift: expected={expected_commit} actual={actual_commit}"
    )
status = git("status", "--porcelain")
if status:
    raise SystemExit(f"recovery source became dirty:\n{status}")
changed_paths = sorted(
    line
    for line in git("diff", "--name-only", parent_commit, expected_commit).splitlines()
    if line
)
expected_paths = sorted(
    [
        "dspark_anchor_scan_recover_eval.sh",
        "scripts/check_block_anchor_checkpoint_attach.py",
        "scripts/eval_block_anchor_nll.py",
        "scripts/finalize_block_anchor_receipt.py",
        "tests/test_block_anchor_campaign.py",
        "tests/test_block_anchor_recovery.py",
    ]
)
if changed_paths != expected_paths:
    raise SystemExit(
        f"recovery source diff escaped allowlist: actual={changed_paths} "
        f"expected={expected_paths}"
    )
payload = {
    "schema": "block_anchor_eval_recovery_source_lock_v1",
    "source_root": str(root),
    "source_commit": actual_commit,
    "source_tree": git("rev-parse", "HEAD^{tree}"),
    "source_author": git("show", "-s", "--format=%an <%ae>", "HEAD"),
    "parent_source_commit": parent_commit,
    "changed_paths": changed_paths,
    "working_tree_clean": True,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("BLOCK_ANCHOR_RECOVERY_SOURCE_OK " + json.dumps(payload, sort_keys=True))
PY
}

# Validate all immutable parent evidence before claiming the one-shot recovery
# output root. The deterministic snapshot is then checked again inside the
# result root and once more immediately before receipt finalization.
preflight_snapshot=$(mktemp "${TMPDIR:-/tmp}/block-anchor-parent.XXXXXXXX.json")
validate_parent "$preflight_snapshot"
source_preflight_snapshot=$(mktemp "${TMPDIR:-/tmp}/block-anchor-source.XXXXXXXX.json")
validate_source "$source_preflight_snapshot"

output_root=$(realpath -m -- "${OUTPUT_ROOT:-/opt/gpfs/users/shuai/experiments/dspark-anchor-seed0-300proof-recovery-20260719}")
case "$output_root" in
  /opt/gpfs/users/shuai/experiments/dspark-anchor-*) ;;
  *) echo "OUTPUT_ROOT must be a new Shuai dspark-anchor experiment: $output_root" >&2; exit 2 ;;
esac
[[ $output_root != "$parent_root" ]] || { echo "recovery root must differ from parent root" >&2; exit 2; }
mkdir "$output_root" || {
  echo "refusing non-new OUTPUT_ROOT: $output_root" >&2
  exit 1
}

log_dir="$output_root/logs"
report_dir="$output_root/reports"
mkdir "$log_dir" "$report_dir"
manifest="$output_root/${run_id}.manifest.txt"
parent_preflight_report="$report_dir/${run_id}.parent_preflight.json"
parent_postflight_report="$report_dir/${run_id}.parent_postflight.json"
source_preflight_report="$report_dir/${run_id}.source_preflight.json"
source_postflight_report="$report_dir/${run_id}.source_postflight.json"
regenerated_training_report="$report_dir/${run_id}.training.json"
regenerated_training_log="$log_dir/${run_id}.training_regeneration.log"
contract_report="$report_dir/${run_id}.contract.log"
contract_post_report="$report_dir/${run_id}.contract_post.log"
source_report="$report_dir/${run_id}.source.log"
vocab_report="$report_dir/${run_id}.vocab.log"
attach_report="$report_dir/${run_id}.attach.log"
environment_report="$report_dir/${run_id}.environment.json"
gpu_report="$report_dir/${run_id}.gpu.csv"
nll_report="$report_dir/${run_id}.nll.json"
nll_log="$log_dir/${run_id}.nll.log"
verdict_report="$report_dir/${run_id}.verdict.json"
receipt_report="$report_dir/${run_id}.receipt.json"

validate_parent "$parent_preflight_report"
cmp -s "$preflight_snapshot" "$parent_preflight_report" || {
  echo "parent evidence changed between preflight and output-root creation" >&2
  exit 1
}
validate_source "$source_preflight_report"
cmp -s "$source_preflight_snapshot" "$source_preflight_report" || {
  echo "source evidence changed between preflight and output-root creation" >&2
  exit 1
}

{
  echo "run_id=$run_id"
  echo "scope=seed0_300step_eval_only_recovery"
  echo "recovery_scope=eval_only"
  echo "training_reused=1"
  echo "training_reexecuted=0"
  echo "seed_index=0"
  echo "train_seed=42"
  echo "eval_seed=20260719"
  echo "english_only_mechanism_probe=1"
  echo "source_root=$root"
  echo "source_commit=$source_commit"
  echo "source_author=$(git show -s --format='%an <%ae>' HEAD)"
  echo "parent_root=$parent_root"
  echo "parent_run_id=$parent_run_id"
  echo "parent_source_commit=$expected_parent_source_commit"
  echo "parent_manifest=$parent_manifest"
  echo "parent_manifest_sha256=$expected_parent_manifest_sha256"
  echo "parent_training_report=$parent_training_report"
  echo "parent_training_report_sha256=$expected_parent_training_sha256"
  echo "parent_failed_phase=nll"
  echo "parent_infrastructure_rc=1"
  echo "parent_failure_log=$parent_failure_log"
  echo "parent_failure_log_sha256=$expected_parent_failure_log_sha256"
  echo "base_checkpoint=$base_checkpoint"
  echo "causal_checkpoint=$parent_causal_checkpoint"
  echo "stateless_checkpoint=$parent_stateless_checkpoint"
  echo "causal_replay_checkpoint=$parent_replay_checkpoint"
  echo "automatic_followup_submitted=0"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$manifest"

phase=source
python - "$root" <<'PY' | tee "$source_report"
import pathlib
import sys
import omnivoice

expected = pathlib.Path(sys.argv[1]).resolve()
actual = pathlib.Path(omnivoice.__file__).resolve()
print(f"python={sys.executable}")
print(f"omnivoice={actual}")
if expected not in actual.parents:
    raise SystemExit(f"refusing non-Shuai source: {actual}")
PY

phase=environment
python - "$source_commit" "$expected_parent_source_commit" "$environment_report" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import accelerate
import liger_kernel
import torch
import transformers

commit, parent_commit, output = sys.argv[1:]
gpu_rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()
driver_rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    text=True,
).strip().splitlines()
liger_root = Path(liger_kernel.__file__).resolve().parent
liger_files = sorted(
    path for path in liger_root.rglob("*")
    if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
)
digest = hashlib.sha256()
for path in liger_files:
    digest.update(path.relative_to(liger_root).as_posix().encode())
    digest.update(b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
payload = {
    "source_commit": commit,
    "parent_source_commit": parent_commit,
    "recovery_scope": "eval_only",
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "transformers_version": transformers.__version__,
    "accelerate_version": accelerate.__version__,
    "liger_kernel_source_root": str(liger_root),
    "liger_kernel_file_count": len(liger_files),
    "liger_kernel_tree_sha256": digest.hexdigest(),
    "nvidia_driver_version": sorted(set(driver_rows)),
    "gpu_inventory": gpu_rows,
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("BLOCK_ANCHOR_RECOVERY_ENVIRONMENT_OK " + json.dumps(payload, sort_keys=True))
PY

phase=gpu
gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $(printf '%s\n' "$gpu_names" | sed '/^$/d' | wc -l) -eq 1 ]] || {
  echo "eval recovery expected exactly one visible GPU" >&2
  exit 1
}
printf '%s\n' "$gpu_names" | grep -vi H100 >/dev/null && exit 1 || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee "$gpu_report"

phase=contract
python scripts/check_block_anchor_contract.py \
  --causal-config "$causal_config" \
  --stateless-config "$stateless_config" \
  --checkpoint "$base_checkpoint" \
  --manifest "$main_manifest" \
  --base-train-config "$base_train_config" \
  --data-config "$data_config" \
  --expected-main-source-commit "$main_source_commit" 2>&1 | tee "$contract_report"
data_manifest_inventory_sha256=$(python - "$contract_report" "$parent_manifest" <<'PY'
import json
import sys

prefix = "BLOCK_ANCHOR_CONTRACT_OK "
lines = [line for line in open(sys.argv[1]) if line.startswith(prefix)]
if len(lines) != 1:
    raise SystemExit(f"expected exactly one contract marker, got {len(lines)}")
actual = json.loads(lines[0][len(prefix):])["data_manifest_inventory_sha256"]
parent = [line.split("=", 1)[1].strip() for line in open(sys.argv[2]) if line.startswith("data_manifest_inventory_sha256=")]
if parent != [actual]:
    raise SystemExit(f"recovery data inventory differs from parent: parent={parent} actual={actual}")
print(actual)
PY
)
echo "data_manifest_inventory_sha256=$data_manifest_inventory_sha256" >> "$manifest"

phase=vocab
python scripts/check_checkpoint_vocab.py --train-config "$causal_config" 2>&1 | tee "$vocab_report"

# This is a real full packed CUDA forward, not a config-only check. It is the
# fail-fast guard for the bf16 FlexAttention q/k/v contract that broke v1 NLL.
phase=attach_cuda_smoke
python scripts/check_block_anchor_checkpoint_attach.py \
  --causal-config "$causal_config" --checkpoint "$base_checkpoint" \
  2>&1 | tee "$attach_report"
grep -q "full_model_synthetic_packed_forward=PASS" "$attach_report" || {
  echo "real CUDA attach smoke did not prove the full packed forward" >&2
  exit 1
}

phase=training_report_regeneration
python scripts/report_block_anchor_training.py \
  --causal-log "$parent_causal_log" --stateless-log "$parent_stateless_log" \
  --causal-replay-log "$parent_replay_log" \
  --causal-memory "$parent_causal_memory" --stateless-memory "$parent_stateless_memory" \
  --causal-replay-memory "$parent_replay_memory" \
  --base-checkpoint "$base_checkpoint" --causal-checkpoint "$parent_causal_checkpoint" \
  --stateless-checkpoint "$parent_stateless_checkpoint" \
  --causal-replay-checkpoint "$parent_replay_checkpoint" \
  --output "$regenerated_training_report" 2>&1 | tee "$regenerated_training_log"
cmp -s "$parent_training_report" "$regenerated_training_report" || {
  echo "regenerated training report is not byte-identical to the locked parent report" >&2
  exit 1
}
echo "training_report_reused_byte_exact=1" >> "$manifest"

phase=nll
python scripts/eval_block_anchor_nll.py \
  --causal-config "$causal_config" --stateless-config "$stateless_config" \
  --base-checkpoint "$base_checkpoint" --causal-checkpoint "$parent_causal_checkpoint" \
  --stateless-checkpoint "$parent_stateless_checkpoint" --num-packs 32 \
  --benchmark-packs 4 --benchmark-warmup 2 --benchmark-repeats 5 \
  --output "$nll_report" 2>&1 | tee "$nll_log"

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
import json
import sys

prefix = "BLOCK_ANCHOR_CONTRACT_OK "
lines = [line for line in open(sys.argv[1]) if line.startswith(prefix)]
if len(lines) != 1:
    raise SystemExit(f"expected exactly one post-contract marker, got {len(lines)}")
print(json.loads(lines[0][len(prefix):])["data_manifest_inventory_sha256"])
PY
)
[[ $post_data_manifest_inventory_sha256 == "$data_manifest_inventory_sha256" ]] || {
  echo "data manifest content drifted during recovery" >&2
  exit 1
}
echo "post_data_manifest_inventory_sha256=$post_data_manifest_inventory_sha256" >> "$manifest"

phase=parent_postflight
validate_parent "$parent_postflight_report"
cmp -s "$parent_preflight_report" "$parent_postflight_report" || {
  echo "locked parent evidence changed during eval recovery" >&2
  exit 1
}
echo "parent_evidence_postflight_exact=1" >> "$manifest"

phase=source_postflight
validate_source "$source_postflight_report"
cmp -s "$source_preflight_report" "$source_postflight_report" || {
  echo "frozen recovery source changed during evaluation" >&2
  exit 1
}
echo "source_evidence_postflight_exact=1" >> "$manifest"

phase=verdict
python - "$regenerated_training_report" "$nll_report" "$verdict_report" <<'PY'
import json
import sys
from pathlib import Path

from scripts.finalize_block_anchor_receipt import resolve_final_verdict

training_path, nll_path, output = map(Path, sys.argv[1:])
training = json.loads(training_path.read_text())
nll = json.loads(nll_path.read_text())
verdict = resolve_final_verdict(training["verdict"], nll["verdict"])
allowed = {
    "PROMOTE_TO_3SEED",
    "SCIENTIFIC_KILL",
    "INCONCLUSIVE_1K",
    "INVALID_IMPLEMENTATION",
    "ENGINEERING_BLOCK",
}
if verdict not in allowed:
    raise SystemExit(f"invalid final verdict: {verdict}")
report = {
    "verdict": verdict,
    "training_verdict": training["verdict"],
    "nll_verdict": nll["verdict"],
    "generation_status": "NEEDS_GENERATION",
    "automatic_followup_submitted": False,
    "scope": "seed-0 300-step eval-only recovery",
    "recovery_scope": "eval_only",
    "training_reused": True,
    "training_reexecuted": False,
    "seed_index": 0,
    "train_seed": 42,
    "eval_seed": 20260719,
    "english_only_mechanism_probe": True,
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print("BLOCK_ANCHOR_RECOVERY_FINAL_VERDICT " + json.dumps(report, sort_keys=True))
PY

scientific_verdict=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$verdict_report")
phase=receipt
python scripts/finalize_block_anchor_receipt.py \
  --manifest "$manifest" --output "$receipt_report" \
  --scientific-verdict "$scientific_verdict" \
  --verdict-report "$verdict_report" \
  --training-report "$regenerated_training_report" \
  --nll-report "$nll_report" \
  --environment-report "$environment_report" \
  --receipt-mode recovery_eval_only \
  --parent-manifest "$parent_manifest" \
  --parent-failure-log "$parent_failure_log" \
  --seed-index 0 --train-seed 42 --eval-seed 20260719 \
  --english-only-mechanism-probe \
  --checkpoint "base=$base_checkpoint" \
  --checkpoint "causal=$parent_causal_checkpoint" \
  --checkpoint "stateless=$parent_stateless_checkpoint" \
  --checkpoint "causal_replay=$parent_replay_checkpoint" \
  --artifact "causal_config=$causal_config" \
  --artifact "stateless_config=$stateless_config" \
  --artifact "base_train_config=$base_train_config" \
  --artifact "data_config=$data_config" \
  --artifact "main_manifest=$main_manifest" \
  --artifact "base_model=$base_checkpoint/model.safetensors" \
  --artifact "parent_manifest=$parent_manifest" \
  --artifact "parent_training_report=$parent_training_report" \
  --artifact "parent_failure_log=$parent_failure_log" \
  --artifact "parent_preflight_report=$parent_preflight_report" \
  --artifact "parent_postflight_report=$parent_postflight_report" \
  --artifact "source_preflight_report=$source_preflight_report" \
  --artifact "source_postflight_report=$source_postflight_report" \
  --artifact "parent_causal_log=$parent_causal_log" \
  --artifact "parent_stateless_log=$parent_stateless_log" \
  --artifact "parent_causal_replay_log=$parent_replay_log" \
  --artifact "parent_causal_memory=$parent_causal_memory" \
  --artifact "parent_stateless_memory=$parent_stateless_memory" \
  --artifact "parent_causal_replay_memory=$parent_replay_memory" \
  --artifact "parent_causal_model=$parent_causal_checkpoint/model.safetensors" \
  --artifact "parent_stateless_model=$parent_stateless_checkpoint/model.safetensors" \
  --artifact "parent_causal_replay_model=$parent_replay_checkpoint/model.safetensors" \
  --artifact "contract_report=$contract_report" \
  --artifact "contract_post_report=$contract_post_report" \
  --artifact "source_report=$source_report" \
  --artifact "vocab_report=$vocab_report" \
  --artifact "attach_report=$attach_report" \
  --artifact "environment_report=$environment_report" \
  --artifact "gpu_report=$gpu_report" \
  --artifact "training_regeneration_log=$regenerated_training_log" \
  --artifact "regenerated_training_report=$regenerated_training_report" \
  --artifact "nll_log=$nll_log" \
  --artifact "nll_report=$nll_report" \
  --artifact "verdict_report=$verdict_report"

echo "BLOCK_ANCHOR_RECOVERY_RESULT verdict=$scientific_verdict generation=NEEDS_GENERATION receipt=$receipt_report"
echo "BLOCK_ANCHOR_RECOVERY_NO_AUTOMATIC_FOLLOWUP"
{
  echo "finished_utc=$(date -u +%FT%TZ)"
  echo "proof_complete=1"
  echo "train_rc=not_run"
  echo "tee_rc=0"
  echo "infrastructure_rc=0"
  echo "rc=0"
} >> "$manifest"
finished=1
echo "BLOCK_ANCHOR_RECOVERY_EXIT run_id=$run_id rc=0 time=$(date -u +%FT%TZ)"
