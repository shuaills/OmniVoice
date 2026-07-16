#!/usr/bin/env bash
# Self-terminating three-arm EOS/CFG score-calibration probe for OMS.
# Every arm uses the same checkpoint, ordered subset, random seed, prompt,
# shared-reference CFG geometry, and decode settings.  Only EOS score
# calibration changes: legacy, renorm, or mass_preserving.
set -Eeuo pipefail

C=${C:-/opt/gpfs/users/shuai/work/training-contract-probes/OmniVoice}
MAIN=${MAIN:-/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice}
E=${E:-/opt/gpfs/users/shuai/work/block-emilia-parity/OmniVoice}
L=${L:-/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice}
DL=${DL:-/opt/gpfs/users/yinfeng/work/OmniVoice}
BASE=${BASE:-/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block}
MODELS=${MODELS:-$DL/download/tts_eval_models}
CK=${CK:-$L/exp/blockcausal_splitloss_emilia_300k_lx20/checkpoint-300000}
CONFIG_SRC=${CONFIG_SRC:-$MAIN/results_scratch_eosdecouple_50k/shim_ckpt/config.json}
TREND=${TREND:-$MAIN/results_wer_trend}

EXPECTED_COUNT=${EXPECTED_COUNT:-100}
MAX_SOURCE_COUNT=${MAX_SOURCE_COUNT:-300}
LANGS=${LANGS:-zh,en}
UTT_IDS=${UTT_IDS:-}
UTT_REGEX=${UTT_REGEX:-}
GPU_IDS=${GPU_IDS:-0,1}
EOS_CFG_TRACE=${EOS_CFG_TRACE:-0}
ITEM_ERROR_POLICY=${ITEM_ERROR_POLICY:-fail-at-end}
RESULT_ROOT=${RESULT_ROOT:-/opt/gpfs/users/shuai/work/training-contract-probes/results/eos-cfg-calibration-first${EXPECTED_COUNT}}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${OMS_JOB_ID:-${HOSTNAME:-host}-$$}}

# Locked controls.  They are constants rather than overridable defaults so an
# arm cannot silently drift away from the intended single-variable test.
GUIDANCE_SCALE=2.0
CFG_UNCONDITIONAL_SEED_POLICY=shared
STEPS_PER_BLOCK=16
BLOCK_SIZE=32
MAX_BLOCKS=24
SILENCE_MATCH_CODEBOOKS=2
PROMPT_CONTRACT=current
LANG_POLICY=dataset
SEED_BASE=20260707

ARM_NAMES=(legacy renorm mass_preserving)
ARM_EOS_CFG_CALIBRATIONS=(legacy renorm mass_preserving)
BASELINE_ARM=legacy

RES=$RESULT_ROOT/$RUN_ID
SHIM=$RES/shim
VERDICT=$RES/VERDICT.txt
PROVENANCE=$RES/PROVENANCE.txt
REPORTER=$C/scripts/training_contract_probe_report.py
GENERATOR=$C/tests/seedtts_blockwise_gen.py
PIDS=()

die() {
  echo "ERROR: $*" >&2
  return 1
}

require_file() {
  [[ -f $1 ]] || die "required file not found: $1"
}

require_dir() {
  [[ -d $1 ]] || die "required directory not found: $1"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if ((${#PIDS[@]})); then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  if ((rc != 0)) && [[ -n ${VERDICT:-} && -e ${RES:-} ]]; then
    echo "EOS_CFG_CALIBRATION_PROBE_FAILED rc=$rc time=$(date -u +%FT%TZ)" \
      | tee -a "$VERDICT" >&2
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_group() {
  local label=$1
  local failed=0
  local pid
  for pid in "${PIDS[@]}"; do
    wait "$pid" || failed=1
  done
  PIDS=()
  ((failed == 0)) || die "$label failed"
}

source_tsv_for_lang() {
  case "$1" in
    zh) echo "$TREND/first300.tsv" ;;
    en) echo "$E/results_autoeval/en_first300.tsv" ;;
    *) die "unsupported language: $1" ;;
  esac
}

source_jsonl_for_lang() {
  case "$1" in
    zh) echo "$TREND/first300.jsonl" ;;
    en) echo "$E/results_autoeval/en_first300.jsonl" ;;
    *) die "unsupported language: $1" ;;
  esac
}

anchor_for_lang() {
  echo "$E/results_endpoint/r1_$1"
}

[[ $RUN_ID =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid RUN_ID: $RUN_ID"
[[ $EXPECTED_COUNT =~ ^[0-9]+$ ]] || die "EXPECTED_COUNT must be an integer"
((EXPECTED_COUNT > 0 && EXPECTED_COUNT <= MAX_SOURCE_COUNT)) || \
  die "EXPECTED_COUNT must be in [1, $MAX_SOURCE_COUNT], got $EXPECTED_COUNT"
[[ -z $UTT_IDS || -z $UTT_REGEX ]] || \
  die "UTT_IDS and UTT_REGEX are mutually exclusive"
case "$EOS_CFG_TRACE" in
  0) TRACE_ARGS=() ;;
  1) TRACE_ARGS=(--eos-cfg-trace) ;;
  *) die "EOS_CFG_TRACE must be 0 or 1, got $EOS_CFG_TRACE" ;;
esac

IFS=, read -r -a LANG_ARRAY <<< "$LANGS"
((${#LANG_ARRAY[@]} > 0)) || die "LANGS must select at least one language"
for ((i = 0; i < ${#LANG_ARRAY[@]}; i++)); do
  lang=${LANG_ARRAY[$i]}
  [[ $lang == zh || $lang == en ]] || die "LANGS supports only zh,en; got $lang"
  for ((j = 0; j < i; j++)); do
    [[ $lang != "${LANG_ARRAY[$j]}" ]] || die "duplicate language in LANGS: $lang"
  done
done
EXPECTED_LANGUAGES=$(IFS=,; echo "${LANG_ARRAY[*]}")

if [[ -n $UTT_IDS ]]; then
  IFS=, read -r -a UTT_ID_ARRAY <<< "$UTT_IDS"
  ((${#UTT_ID_ARRAY[@]} == EXPECTED_COUNT)) || \
    die "UTT_IDS count must equal EXPECTED_COUNT: ids=${#UTT_ID_ARRAY[@]} count=$EXPECTED_COUNT"
fi

[[ ${#ARM_NAMES[@]} -eq 3 ]] || die "EOS calibration matrix must contain three arms"
[[ ${#ARM_EOS_CFG_CALIBRATIONS[@]} -eq ${#ARM_NAMES[@]} ]] || \
  die "EOS calibration matrix length mismatch"

IFS=, read -r -a GPU_ARRAY <<< "$GPU_IDS"
((${#GPU_ARRAY[@]} >= 2)) || die "GPU_IDS must provide at least two GPUs"
for ((i = 0; i < ${#GPU_ARRAY[@]}; i++)); do
  [[ ${GPU_ARRAY[$i]} =~ ^[0-9]+$ ]] || die "invalid GPU ID: ${GPU_ARRAY[$i]}"
  for ((j = 0; j < i; j++)); do
    [[ ${GPU_ARRAY[$i]} != "${GPU_ARRAY[$j]}" ]] || \
      die "duplicate GPU ID: ${GPU_ARRAY[$i]}"
  done
done

require_dir "$C"
require_dir "$MAIN"
require_dir "$E"
require_dir "$L"
require_dir "$DL"
require_dir "$BASE"
require_dir "$BASE/audio_tokenizer"
require_dir "$MODELS"
require_dir "$CK"
require_file "$DL/.venv/bin/activate"
require_file "$CONFIG_SRC"
require_file "$REPORTER"
require_file "$C/scripts/decode_sweep_report.py"
require_file "$GENERATOR"
require_file "$C/omnivoice/blockdiff.py"
require_file "$C/omnivoice/blockdiff_dual.py"
require_file "$C/omnivoice/eval/seedtts_blockwise_contract.py"
require_file "$C/omnivoice/eval/wer/seedtts.py"
require_file "$C/omnivoice/eval/speaker_similarity/sim.py"
require_file "$CK/model.safetensors"
for lang in "${LANG_ARRAY[@]}"; do
  require_file "$(source_tsv_for_lang "$lang")"
  require_file "$(source_jsonl_for_lang "$lang")"
  require_dir "$(anchor_for_lang "$lang")"
done

mkdir -p "$RESULT_ROOT"
mkdir "$RES"
mkdir "$RES/logs" "$RES/inputs" "$RES/filtered_sources" "$RES/arms" "$SHIM"
: > "$VERDICT"
v() { echo "$*" | tee -a "$VERDICT"; }

cd "$C"
# shellcheck disable=SC1090
source "$DL/.venv/bin/activate"
export PYTHONPATH=$C
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

git diff --quiet --no-ext-diff || die "tracked unstaged changes make provenance ambiguous"
git diff --cached --quiet --no-ext-diff || die "staged changes make provenance ambiguous"
for tracked in \
  training_contract_eos_cfg_calibration_probe.sh \
  scripts/training_contract_probe_report.py \
  scripts/decode_sweep_report.py \
  tests/seedtts_blockwise_gen.py \
  omnivoice/blockdiff.py \
  omnivoice/blockdiff_dual.py \
  omnivoice/eval/seedtts_blockwise_contract.py; do
  git ls-files --error-unmatch "$tracked" >/dev/null 2>&1 || \
    die "executed code is not tracked: $tracked"
done

commit=$(git rev-parse HEAD)
branch=$(git branch --show-current)
ck_real=$(realpath "$CK")
base_real=$(realpath "$BASE")
config_real=$(realpath "$CONFIG_SRC")
v "EOS_CFG_CALIBRATION_PROBE_START run_id=$RUN_ID commit=$commit time=$(date -u +%FT%TZ)"
v "MATRIX arms=${ARM_NAMES[*]} langs=$EXPECTED_LANGUAGES count=$EXPECTED_COUNT checkpoint=$ck_real gpu_ids=$GPU_IDS"
v "FIXED prompt=$PROMPT_CONTRACT cfg_seed=$CFG_UNCONDITIONAL_SEED_POLICY guidance=$GUIDANCE_SCALE steps=$STEPS_PER_BLOCK block_size=$BLOCK_SIZE max_blocks=$MAX_BLOCKS trace=$EOS_CFG_TRACE"

cp "$config_real" "$SHIM/config.json"
ln -s "$ck_real/model.safetensors" "$SHIM/model.safetensors"
for file in tokenizer.json tokenizer_config.json train_config.json chat_template.jinja; do
  if [[ -f "$ck_real/$file" ]]; then
    ln -s "$ck_real/$file" "$SHIM/$file"
  fi
done

# Optional exact-id or regex filtering supports a one/few-utterance microprobe
# without changing the canonical source files.  The filter is fail-closed and
# preserves paired TSV/JSONL order before the usual manifest builder runs.
for lang in "${LANG_ARRAY[@]}"; do
  source_tsv=$(source_tsv_for_lang "$lang")
  source_jsonl=$(source_jsonl_for_lang "$lang")
  if [[ -n $UTT_IDS || -n $UTT_REGEX ]]; then
    filtered_dir=$RES/filtered_sources/$lang
    mkdir "$filtered_dir"
    python - "$source_tsv" "$source_jsonl" "$filtered_dir" \
      "$EXPECTED_COUNT" "$UTT_IDS" "$UTT_REGEX" <<'PY'
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

source_tsv, source_jsonl, output_dir = map(Path, sys.argv[1:4])
source_tsv = source_tsv.resolve()
source_jsonl = source_jsonl.resolve()
expected = int(sys.argv[4])
exact_ids = [value for value in sys.argv[5].split(",") if value]
pattern = re.compile(sys.argv[6]) if sys.argv[6] else None

with source_tsv.open(newline="", encoding="utf-8") as handle:
    tsv_rows = [row for row in csv.reader(handle, delimiter="\t") if row]
with source_jsonl.open(encoding="utf-8") as handle:
    json_rows = [json.loads(line) for line in handle if line.strip()]
if [row[0] for row in tsv_rows] != [str(row.get("id")) for row in json_rows]:
    raise SystemExit("source TSV/JSONL IDs are not paired in the same order")

indexed_pairs = [
    (index, tsv_row, json_row)
    for index, (tsv_row, json_row) in enumerate(zip(tsv_rows, json_rows))
]
pairs = {tsv_row[0]: (index, tsv_row, json_row)
         for index, tsv_row, json_row in indexed_pairs}
if exact_ids:
    missing = [utt_id for utt_id in exact_ids if utt_id not in pairs]
    if missing:
        raise SystemExit(f"UTT_IDS missing from {source_tsv}: {missing}")
    selected = [pairs[utt_id] for utt_id in exact_ids]
else:
    selected = [
        (index, tsv_row, json_row)
        for index, tsv_row, json_row in indexed_pairs
        if pattern.search(tsv_row[0])
    ][:expected]
if len(selected) != expected:
    raise SystemExit(
        f"filter selected {len(selected)} rows from {source_tsv}; expected {expected}"
    )

with (output_dir / "source.tsv").open("w", newline="", encoding="utf-8") as handle:
    csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(
        tsv_row for _, tsv_row, _ in selected
    )
with (output_dir / "source.jsonl").open("w", encoding="utf-8") as handle:
    for _, _, json_row in selected:
        handle.write(json.dumps(json_row, ensure_ascii=False) + "\n")
seed_index_map = {
    tsv_row[0]: index for index, tsv_row, _ in selected
}
(output_dir / "seed_index_map.json").write_text(
    json.dumps(seed_index_map, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
selection_manifest = {
    "canonical_source_tsv": str(source_tsv),
    "canonical_source_tsv_sha256": hashlib.sha256(
        source_tsv.read_bytes()
    ).hexdigest(),
    "canonical_source_jsonl": str(source_jsonl),
    "canonical_source_jsonl_sha256": hashlib.sha256(
        source_jsonl.read_bytes()
    ).hexdigest(),
    "selection": [
        {"id": tsv_row[0], "source_index": index}
        for index, tsv_row, _ in selected
    ],
}
(output_dir / "selection_manifest.json").write_text(
    json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
(output_dir / "prompt_wavs").symlink_to(
    source_tsv.parent / "prompt_wavs", target_is_directory=True
)
PY
    source_tsv=$filtered_dir/source.tsv
    source_jsonl=$filtered_dir/source.jsonl
  fi
  python "$REPORTER" prepare-inputs \
    --source-tsv "$source_tsv" \
    --source-jsonl "$source_jsonl" \
    --count "$EXPECTED_COUNT" \
    --output-dir "$RES/inputs/$lang" \
    --anchor-dir "$(anchor_for_lang "$lang")" \
    --check-ref-audio | tee -a "$VERDICT"
done

{
  echo "run=eos_cfg_calibration_probe"
  echo "created_utc=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname -f 2>/dev/null || hostname)"
  echo "oms_job_id=${OMS_JOB_ID:-unset}"
  echo "repo=$(realpath "$C")"
  echo "branch=$branch"
  echo "commit=$commit"
  echo "git_tracked_status=clean"
  echo "git_status_begin"
  git status --porcelain=v1 --untracked-files=all
  echo "git_status_end"
  echo "checkpoint=$ck_real"
  echo "config_source=$config_real"
  echo "base_model=$base_real"
  echo "expected_count=$EXPECTED_COUNT"
  echo "languages=$EXPECTED_LANGUAGES"
  echo "utt_ids=${UTT_IDS:-unset}"
  echo "utt_regex=${UTT_REGEX:-unset}"
  echo "gpu_ids=$GPU_IDS"
  echo "trace=$EOS_CFG_TRACE"
  echo "generation_shards=${#GPU_ARRAY[@]}"
  echo "seed_contract=torch.manual_seed($SEED_BASE + generation_seed_index); filtered probes preserve canonical source row indices; identical subset and shard count for every arm"
  echo "fixed_contract=prompt=$PROMPT_CONTRACT cfg_seed=$CFG_UNCONDITIONAL_SEED_POLICY guidance=$GUIDANCE_SCALE steps=$STEPS_PER_BLOCK block_size=$BLOCK_SIZE max_blocks=$MAX_BLOCKS dtype=bf16 silence_stop=0"
  echo "arms=${ARM_NAMES[*]}"
  echo "code_sha256_begin"
  sha256sum \
    training_contract_eos_cfg_calibration_probe.sh \
    scripts/training_contract_probe_report.py \
    scripts/decode_sweep_report.py \
    tests/seedtts_blockwise_gen.py \
    omnivoice/blockdiff.py \
    omnivoice/blockdiff_dual.py \
    omnivoice/eval/seedtts_blockwise_contract.py \
    omnivoice/models/omnivoice.py
  echo "code_sha256_end"
  echo "checkpoint_sha256_begin"
  sha256sum "$ck_real/model.safetensors" "$config_real"
  for file in tokenizer.json tokenizer_config.json train_config.json chat_template.jinja; do
    [[ ! -f "$ck_real/$file" ]] || sha256sum "$ck_real/$file"
  done
  echo "checkpoint_sha256_end"
  echo "base_model_sha256_begin"
  find "$base_real" \( -type f -o -type l \) -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 -r sha256sum
  echo "base_model_sha256_end"
  echo "evaluation_models_sha256_begin"
  find "$MODELS" \( -type f -o -type l \) -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 -r sha256sum
  echo "evaluation_models_sha256_end"
  echo "input_manifest_sha256_begin"
  for lang in "${LANG_ARRAY[@]}"; do
    sha256sum "$RES/inputs/$lang/manifest.json"
  done
  echo "input_manifest_sha256_end"
  echo "filtered_selection_manifest_begin"
  for lang in "${LANG_ARRAY[@]}"; do
    selection_manifest=$RES/filtered_sources/$lang/selection_manifest.json
    if [[ -f $selection_manifest ]]; then
      sha256sum "$selection_manifest"
      cat "$selection_manifest"
      seed_index_map=$RES/filtered_sources/$lang/seed_index_map.json
      sha256sum "$seed_index_map"
      cat "$seed_index_map"
    fi
  done
  echo "filtered_selection_manifest_end"
  echo "python=$($DL/.venv/bin/python --version 2>&1)"
  "$DL/.venv/bin/python" - <<'PY'
import torch
import transformers
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
PY
  uname -a
  nvidia-smi -L
} > "$PROVENANCE"

for ((arm_index = 0; arm_index < ${#ARM_NAMES[@]}; arm_index++)); do
  arm=${ARM_NAMES[$arm_index]}
  eos_calibration=${ARM_EOS_CFG_CALIBRATIONS[$arm_index]}
  for lang in "${LANG_ARRAY[@]}"; do
    tsv=$RES/inputs/$lang/test.tsv
    jsonl=$RES/inputs/$lang/test.jsonl
    anchor=$(anchor_for_lang "$lang")
    arm_dir=$RES/arms/$arm/$lang
    wav_dir=$arm_dir/wavs
    seed_index_map=$RES/filtered_sources/$lang/seed_index_map.json
    SEED_INDEX_ARGS=()
    if [[ -f $seed_index_map ]]; then
      SEED_INDEX_ARGS=(--generation-seed-index-map "$seed_index_map")
    fi
    mkdir -p "$arm_dir/logs"
    mkdir "$wav_dir"
    v "ARM_START arm=$arm lang=$lang eos_cfg_calibration=$eos_calibration time=$(date -u +%FT%TZ)"

    PIDS=()
    for ((shard = 0; shard < ${#GPU_ARRAY[@]}; shard++)); do
      gpu=${GPU_ARRAY[$shard]}
      : > "$wav_dir/gen_meta_shard${shard}.jsonl"
      : > "$wav_dir/failures_shard${shard}.jsonl"
      CUDA_VISIBLE_DEVICES=$gpu python "$GENERATOR" \
        --tsv "$tsv" \
        --ckpt "$SHIM" \
        --base "$BASE" \
        --out "$wav_dir" \
        --steps-per-block "$STEPS_PER_BLOCK" \
        --max-blocks "$MAX_BLOCKS" \
        --block-size "$BLOCK_SIZE" \
        --guidance-scale "$GUIDANCE_SCALE" \
        "${SEED_INDEX_ARGS[@]}" \
        --eos-cfg-calibration "$eos_calibration" \
        "${TRACE_ARGS[@]}" \
        --dtype bf16 \
        --prompt-contract "$PROMPT_CONTRACT" \
        --cfg-unconditional-seed-policy "$CFG_UNCONDITIONAL_SEED_POLICY" \
        --item-error-policy "$ITEM_ERROR_POLICY" \
        --silence-stop-seconds 0 \
        --silence-match-codebooks "$SILENCE_MATCH_CODEBOOKS" \
        --shard "$shard/${#GPU_ARRAY[@]}" \
        > "$arm_dir/logs/gen_shard${shard}.log" 2>&1 &
      PIDS+=("$!")
    done
    wait_group "generation arm=$arm lang=$lang"

    python - "$wav_dir" "$eos_calibration" "$EXPECTED_COUNT" \
      "$BLOCK_SIZE" "$EOS_CFG_TRACE" "$tsv" "$SEED_BASE" \
      "${seed_index_map:-}" <<'PY' | tee -a "$VERDICT"
import glob
import json
import sys
from pathlib import Path

wav_dir = Path(sys.argv[1])
policy = sys.argv[2]
expected = int(sys.argv[3])
block_size = int(sys.argv[4])
trace_enabled = sys.argv[5] == "1"
tsv_path = Path(sys.argv[6])
seed_base = int(sys.argv[7])
seed_index_map_path = Path(sys.argv[8]) if sys.argv[8] else None
with tsv_path.open(encoding="utf-8") as handle:
    ordered_ids = [line.split("\t", 1)[0] for line in handle if line.strip()]
expected_seed_indices = {
    utt_id: index for index, utt_id in enumerate(ordered_ids)
}
if seed_index_map_path is not None and seed_index_map_path.is_file():
    expected_seed_indices = json.loads(
        seed_index_map_path.read_text(encoding="utf-8")
    )
rows = []
for path in sorted(glob.glob(str(wav_dir / "gen_meta_shard*.jsonl"))):
    with open(path, encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
if len(rows) != expected:
    raise SystemExit(f"metadata count mismatch: expected={expected} actual={len(rows)}")
for row in rows:
    if row.get("eos_cfg_calibration") != policy:
        raise SystemExit(f"metadata policy mismatch for {row.get('utt_id')}: {row.get('eos_cfg_calibration')}")
    seed_mod = row.get("seed_frames_mod_block")
    first = row.get("first_target_block_frames")
    expected_first = block_size - seed_mod if seed_mod else block_size
    if first != expected_first:
        raise SystemExit(f"seed/block geometry mismatch for {row.get('utt_id')}")
    if trace_enabled != ("eos_cfg_trace" in row):
        raise SystemExit(f"trace presence mismatch for {row.get('utt_id')}")
    if trace_enabled:
        trace = row["eos_cfg_trace"]
        required = {
            "guided_eos_mass", "conditional_eos_mass", "legacy_eos_mass",
            "post_eos_mass", "legacy_total_mass", "post_total_mass",
            "guided_margin", "legacy_margin", "post_margin",
            "queue_cutoff", "queue_rank", "legacy_queue_cutoff",
            "legacy_queue_rank", "selected_eos_cols",
        }
        if not isinstance(trace, list) or not trace:
            raise SystemExit(f"empty or invalid trace for {row.get('utt_id')}")
        if any(not required.issubset(step) for step in trace):
            raise SystemExit(f"incomplete trace step for {row.get('utt_id')}")
    expected_seed_index = expected_seed_indices.get(row.get("utt_id"))
    if row.get("generation_seed_index") != expected_seed_index:
        raise SystemExit(f"seed index mismatch for {row.get('utt_id')}")
    if row.get("generation_seed_value") != seed_base + expected_seed_index:
        raise SystemExit(f"seed value mismatch for {row.get('utt_id')}")
print(f"EOS_CFG_METADATA_OK policy={policy} rows={len(rows)} trace={trace_enabled}")
PY

    SUMMARY_ARGS=()
    [[ $arm != "$BASELINE_ARM" ]] || SUMMARY_ARGS+=(--is-baseline)
    VALIDATE_SEED_ARGS=()
    if [[ -f $seed_index_map ]]; then
      VALIDATE_SEED_ARGS=(--seed-index-map "$seed_index_map")
    fi
    python "$REPORTER" validate-generation \
      --tsv "$tsv" \
      --jsonl "$jsonl" \
      --expected-count "$EXPECTED_COUNT" \
      --num-shards "${#GPU_ARRAY[@]}" \
      --arm "$arm" \
      --lang "$lang" \
      --lang-policy "$LANG_POLICY" \
      --prompt-contract "$PROMPT_CONTRACT" \
      --cfg-unconditional-seed-policy "$CFG_UNCONDITIONAL_SEED_POLICY" \
      --guidance-scale "$GUIDANCE_SCALE" \
      --seed-base "$SEED_BASE" \
      "${VALIDATE_SEED_ARGS[@]}" \
      --wav-dir "$wav_dir" \
      --output "$arm_dir/generation_audit.json" \
      "${SUMMARY_ARGS[@]}" | tee -a "$VERDICT"

    PIDS=()
    CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} python omnivoice/eval/wer/seedtts.py \
      --wav-path "$wav_dir" \
      --test-list "$jsonl" \
      --model-dir "$MODELS" \
      --lang "$lang" \
      --decode-path "$arm_dir/wer.tsv" \
      --batch-size 4 \
      > "$arm_dir/logs/wer.log" 2>&1 &
    PIDS+=("$!")
    CUDA_VISIBLE_DEVICES=${GPU_ARRAY[1]} python omnivoice/eval/speaker_similarity/sim.py \
      --wav-path "$wav_dir" \
      --test-list "$jsonl" \
      --model-dir "$MODELS" \
      --decode-path "$arm_dir/sim.tsv" \
      > "$arm_dir/logs/sim.log" 2>&1 &
    PIDS+=("$!")
    wait_group "scoring arm=$arm lang=$lang"

    python "$REPORTER" summarize-arm \
      --arm "$arm" \
      --lang "$lang" \
      --lang-policy "$LANG_POLICY" \
      --prompt-contract "$PROMPT_CONTRACT" \
      --cfg-unconditional-seed-policy "$CFG_UNCONDITIONAL_SEED_POLICY" \
      --guidance-scale "$GUIDANCE_SCALE" \
      --steps-per-block "$STEPS_PER_BLOCK" \
      --expected-count "$EXPECTED_COUNT" \
      --tsv "$tsv" \
      --jsonl "$jsonl" \
      --wav-dir "$wav_dir" \
      --anchor-dir "$anchor" \
      --wer-tsv "$arm_dir/wer.tsv" \
      --sim-tsv "$arm_dir/sim.tsv" \
      --meta-glob "$wav_dir/gen_meta_shard*.jsonl" \
      --failures-glob "$wav_dir/failures_shard*.jsonl" \
      --per-utt-out "$arm_dir/per_utt.tsv" \
      --summary-out "$arm_dir/summary.json" \
      "${SUMMARY_ARGS[@]}" | tee -a "$VERDICT"
  done
done

shopt -s nullglob
SUMMARY_FILES=("$RES"/arms/*/*/summary.json)
shopt -u nullglob
expected_summaries=$((${#ARM_NAMES[@]} * ${#LANG_ARRAY[@]}))
((${#SUMMARY_FILES[@]} == expected_summaries)) || \
  die "expected $expected_summaries summaries, got ${#SUMMARY_FILES[@]}"
expected_arms=$(IFS=,; echo "${ARM_NAMES[*]}")
python "$REPORTER" aggregate \
  --summaries "${SUMMARY_FILES[@]}" \
  --expected-arms "$expected_arms" \
  --expected-languages "$EXPECTED_LANGUAGES" \
  --baseline-arm "$BASELINE_ARM" \
  --expected-count "$EXPECTED_COUNT" \
  --output-tsv "$RES/SUMMARY.tsv" \
  --output-json "$RES/SUMMARY.json" \
  --output-md "$RES/SUMMARY.md" | tee -a "$VERDICT"

tee -a "$VERDICT" < "$RES/SUMMARY.md"
v "EOS_CFG_CALIBRATION_PROBE_DONE run_id=$RUN_ID result=$RES time=$(date -u +%FT%TZ)"
