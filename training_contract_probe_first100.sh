#!/usr/bin/env bash
# Self-terminating first-100 (or first-300) training-contract probe for OMS.
# Seven fresh paired arms isolate language, prompt, CFG-reference, and guidance effects.
set -Eeuo pipefail

# Repository and immutable model/data inputs.
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

# Run contract. EXPECTED_COUNT=100 is the pilot; 300 expands the exact same matrix.
EXPECTED_COUNT=${EXPECTED_COUNT:-100}
MAX_SOURCE_COUNT=${MAX_SOURCE_COUNT:-300}
GPU_IDS=${GPU_IDS:-0,1,2}
STEPS_PER_BLOCK=${STEPS_PER_BLOCK:-16}
BLOCK_SIZE=${BLOCK_SIZE:-32}
MAX_BLOCKS=${MAX_BLOCKS:-24}
SILENCE_MATCH_CODEBOOKS=${SILENCE_MATCH_CODEBOOKS:-2}
ITEM_ERROR_POLICY=${ITEM_ERROR_POLICY:-fail-at-end}
RESULT_ROOT=${RESULT_ROOT:-/opt/gpfs/users/shuai/work/training-contract-probes/results/first${EXPECTED_COUNT}}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${OMS_JOB_ID:-${HOSTNAME:-host}-$$}}

# Canonical generator CLI mapping, centralized for interface integration.
PROMPT_CONTRACT_FLAG=${PROMPT_CONTRACT_FLAG:---prompt-contract}
CFG_UNCONDITIONAL_SEED_POLICY_FLAG=${CFG_UNCONDITIONAL_SEED_POLICY_FLAG:---cfg-unconditional-seed-policy}
PROMPT_CURRENT=${PROMPT_CURRENT:-current}
PROMPT_OFFICIAL=${PROMPT_OFFICIAL:-official-emilia}
CFG_SHARED=${CFG_SHARED:-shared}
CFG_DROP_REF=${CFG_DROP_REF:-drop_ref}
LANG_NONE_VALUE=${LANG_NONE_VALUE:-None}

# arm:               baseline lang_none prompt_official eval_parity cfg_drop_ref combined gs0
ARM_NAMES=(baseline lang_none prompt_official eval_parity cfg_drop_ref combined gs0)
ARM_LANG_POLICIES=(dataset none dataset none dataset none dataset)
ARM_PROMPTS=(
  "$PROMPT_CURRENT" "$PROMPT_CURRENT" "$PROMPT_OFFICIAL" "$PROMPT_OFFICIAL"
  "$PROMPT_CURRENT" "$PROMPT_OFFICIAL" "$PROMPT_CURRENT"
)
ARM_CFG_POLICIES=(
  "$CFG_SHARED" "$CFG_SHARED" "$CFG_SHARED" "$CFG_SHARED"
  "$CFG_DROP_REF" "$CFG_DROP_REF" "$CFG_SHARED"
)
ARM_GUIDANCES=(2.0 2.0 2.0 2.0 2.0 2.0 0)
BASELINE_ARM=baseline

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
    echo "TRAINING_CONTRACT_PROBE_FAILED rc=$rc time=$(date -u +%FT%TZ)" \
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
[[ ${#ARM_NAMES[@]} -eq 7 ]] || die "the canonical matrix must contain seven arms"
for array_length in \
  "${#ARM_LANG_POLICIES[@]}" \
  "${#ARM_PROMPTS[@]}" \
  "${#ARM_CFG_POLICIES[@]}" \
  "${#ARM_GUIDANCES[@]}"; do
  [[ $array_length -eq ${#ARM_NAMES[@]} ]] || die "arm matrix array length mismatch"
done
declare -A ARM_SIGNATURES=()
for ((i = 0; i < ${#ARM_NAMES[@]}; i++)); do
  signature="${ARM_LANG_POLICIES[$i]}|${ARM_PROMPTS[$i]}|${ARM_CFG_POLICIES[$i]}|${ARM_GUIDANCES[$i]}"
  [[ -z ${ARM_SIGNATURES[$signature]+x} ]] || \
    die "duplicate arm argv contract: ${ARM_SIGNATURES[$signature]} and ${ARM_NAMES[$i]}"
  ARM_SIGNATURES[$signature]=${ARM_NAMES[$i]}
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
require_file "$C/omnivoice/blockdiff_dual.py"
require_file "$C/omnivoice/eval/wer/seedtts.py"
require_file "$C/omnivoice/eval/speaker_similarity/sim.py"
require_file "$CK/model.safetensors"
for lang in zh en; do
  require_file "$(source_tsv_for_lang "$lang")"
  require_file "$(source_jsonl_for_lang "$lang")"
  require_dir "$(anchor_for_lang "$lang")"
done

IFS=, read -r -a GPU_ARRAY <<< "$GPU_IDS"
((${#GPU_ARRAY[@]} >= 2)) || die "GPU_IDS must provide at least two GPUs"
for ((i = 0; i < ${#GPU_ARRAY[@]}; i++)); do
  [[ ${GPU_ARRAY[$i]} =~ ^[0-9]+$ ]] || die "invalid GPU ID: ${GPU_ARRAY[$i]}"
  for ((j = 0; j < i; j++)); do
    [[ ${GPU_ARRAY[$i]} != "${GPU_ARRAY[$j]}" ]] || \
      die "duplicate GPU ID: ${GPU_ARRAY[$i]}"
  done
done

mkdir -p "$RESULT_ROOT"
mkdir "$RES"
mkdir "$RES/logs" "$RES/inputs" "$RES/arms" "$SHIM"
: > "$VERDICT"
v() { echo "$*" | tee -a "$VERDICT"; }

cd "$C"
# shellcheck disable=SC1090
source "$DL/.venv/bin/activate"
export PYTHONPATH=$C
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Executed code must be committed and tracked. Unrelated untracked GPFS artifacts
# are fingerprinted but do not invalidate an otherwise exact commit.
git diff --quiet --no-ext-diff || die "tracked unstaged changes make provenance ambiguous"
git diff --cached --quiet --no-ext-diff || die "staged changes make provenance ambiguous"
for tracked in \
  training_contract_probe_first100.sh \
  scripts/training_contract_probe_report.py \
  scripts/decode_sweep_report.py \
  tests/seedtts_blockwise_gen.py \
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
v "TRAINING_CONTRACT_PROBE_START run_id=$RUN_ID commit=$commit time=$(date -u +%FT%TZ)"
v "MATRIX arms=${ARM_NAMES[*]} langs=zh,en count=$EXPECTED_COUNT checkpoint=$ck_real gpu_ids=$GPU_IDS"
v "FIXED steps=$STEPS_PER_BLOCK block_size=$BLOCK_SIZE max_blocks=$MAX_BLOCKS dtype=bf16 silence_stop=0"

cp "$config_real" "$SHIM/config.json"
ln -s "$ck_real/model.safetensors" "$SHIM/model.safetensors"
for file in tokenizer.json tokenizer_config.json train_config.json chat_template.jinja; do
  if [[ -f "$ck_real/$file" ]]; then
    ln -s "$ck_real/$file" "$SHIM/$file"
  fi
done

for lang in zh en; do
  python "$REPORTER" prepare-inputs \
    --source-tsv "$(source_tsv_for_lang "$lang")" \
    --source-jsonl "$(source_jsonl_for_lang "$lang")" \
    --count "$EXPECTED_COUNT" \
    --output-dir "$RES/inputs/$lang" \
    --anchor-dir "$(anchor_for_lang "$lang")" \
    --check-ref-audio | tee -a "$VERDICT"
done

{
  echo "run=training_contract_probe"
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
  echo "gpu_ids=$GPU_IDS"
  echo "generation_shards=${#GPU_ARRAY[@]}"
  echo "seed_contract=torch.manual_seed(20260707 + global_subset_row_index); identical ordered subset and shard count for every arm"
  echo "arms_begin"
  for ((i = 0; i < ${#ARM_NAMES[@]}; i++)); do
    echo "${ARM_NAMES[$i]} lang=${ARM_LANG_POLICIES[$i]} prompt=${ARM_PROMPTS[$i]} cfg_seed=${ARM_CFG_POLICIES[$i]} guidance=${ARM_GUIDANCES[$i]}"
  done
  echo "arms_end"
  echo "code_sha256_begin"
  sha256sum \
    training_contract_probe_first100.sh \
    scripts/training_contract_probe_report.py \
    scripts/decode_sweep_report.py \
    tests/seedtts_blockwise_gen.py \
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
  sha256sum "$RES/inputs/zh/manifest.json" "$RES/inputs/en/manifest.json"
  echo "input_manifest_sha256_end"
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
  lang_policy=${ARM_LANG_POLICIES[$arm_index]}
  prompt_contract=${ARM_PROMPTS[$arm_index]}
  cfg_policy=${ARM_CFG_POLICIES[$arm_index]}
  guidance=${ARM_GUIDANCES[$arm_index]}
  for lang in zh en; do
    tsv=$RES/inputs/$lang/test.tsv
    jsonl=$RES/inputs/$lang/test.jsonl
    anchor=$(anchor_for_lang "$lang")
    arm_dir=$RES/arms/$arm/$lang
    wav_dir=$arm_dir/wavs
    mkdir -p "$arm_dir/logs"
    mkdir "$wav_dir"
    v "ARM_START arm=$arm lang=$lang lang_policy=$lang_policy prompt=$prompt_contract cfg_seed=$cfg_policy guidance=$guidance time=$(date -u +%FT%TZ)"

    LANG_ARGS=()
    [[ $lang_policy == dataset ]] || LANG_ARGS+=(--lang "$LANG_NONE_VALUE")
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
        --guidance-scale "$guidance" \
        --dtype bf16 \
        "${LANG_ARGS[@]}" \
        "$PROMPT_CONTRACT_FLAG" "$prompt_contract" \
        "$CFG_UNCONDITIONAL_SEED_POLICY_FLAG" "$cfg_policy" \
        --item-error-policy "$ITEM_ERROR_POLICY" \
        --silence-stop-seconds 0 \
        --silence-match-codebooks "$SILENCE_MATCH_CODEBOOKS" \
        --shard "$shard/${#GPU_ARRAY[@]}" \
        > "$arm_dir/logs/gen_shard${shard}.log" 2>&1 &
      PIDS+=("$!")
    done
    wait_group "generation arm=$arm lang=$lang"

    SUMMARY_ARGS=()
    [[ $arm != "$BASELINE_ARM" ]] || SUMMARY_ARGS+=(--is-baseline)
    python "$REPORTER" validate-generation \
      --tsv "$tsv" \
      --jsonl "$jsonl" \
      --expected-count "$EXPECTED_COUNT" \
      --num-shards "${#GPU_ARRAY[@]}" \
      --arm "$arm" \
      --lang "$lang" \
      --lang-policy "$lang_policy" \
      --prompt-contract "$prompt_contract" \
      --cfg-unconditional-seed-policy "$cfg_policy" \
      --guidance-scale "$guidance" \
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
      --lang-policy "$lang_policy" \
      --prompt-contract "$prompt_contract" \
      --cfg-unconditional-seed-policy "$cfg_policy" \
      --guidance-scale "$guidance" \
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
expected_summaries=$((${#ARM_NAMES[@]} * 2))
((${#SUMMARY_FILES[@]} == expected_summaries)) || \
  die "expected $expected_summaries summaries, got ${#SUMMARY_FILES[@]}"
expected_arms=$(IFS=,; echo "${ARM_NAMES[*]}")
python "$REPORTER" aggregate \
  --summaries "${SUMMARY_FILES[@]}" \
  --expected-arms "$expected_arms" \
  --expected-languages zh,en \
  --baseline-arm "$BASELINE_ARM" \
  --expected-count "$EXPECTED_COUNT" \
  --output-tsv "$RES/SUMMARY.tsv" \
  --output-json "$RES/SUMMARY.json" \
  --output-md "$RES/SUMMARY.md" | tee -a "$VERDICT"

tee -a "$VERDICT" < "$RES/SUMMARY.md"
v "TRAINING_CONTRACT_PROBE_DONE run_id=$RUN_ID result=$RES time=$(date -u +%FT%TZ)"
