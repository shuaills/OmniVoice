#!/usr/bin/env bash
# Self-terminating CFG guidance probe for OMS.
# Calibrate compares the historical shared-reference guidance=2 control with
# drop_ref guidance in {0.5, 1.0, 1.5, 2.0} on first100. Promote reruns only
# that control and one explicitly selected drop_ref candidate on first300.
# Shared_sweep holds the reference geometry fixed and scans guidance from zero
# through the historical 2.0 setting on a first5 smoke or first100 evaluation.
set -Eeuo pipefail

# Repository and immutable model/data inputs. These intentionally match
# training_contract_probe_first100.sh.
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

# Calibrate is the safe default. Promote must be explicitly selected together
# with its candidate and first300 count. Three shards preserve the generator's
# global seed mapping exactly:
# torch.manual_seed(20260707 + global_subset_row_index).
MODE=${MODE:-calibrate}
EXPECTED_COUNT=${EXPECTED_COUNT:-100}
PROMOTED_GUIDANCE=${PROMOTED_GUIDANCE:-}
GPU_IDS=${GPU_IDS:-0,1,2}
STEPS_PER_BLOCK=${STEPS_PER_BLOCK:-16}
BLOCK_SIZE=${BLOCK_SIZE:-32}
MAX_BLOCKS=${MAX_BLOCKS:-24}
SILENCE_MATCH_CODEBOOKS=${SILENCE_MATCH_CODEBOOKS:-2}
ITEM_ERROR_POLICY=${ITEM_ERROR_POLICY:-fail-at-end}
RESULT_ROOT=${RESULT_ROOT:-/opt/gpfs/users/shuai/work/training-contract-probes/results/cfg-guidance-first${EXPECTED_COUNT}}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${OMS_JOB_ID:-${HOSTNAME:-host}-$$}}

PROMPT_CONTRACT=current
LANG_POLICY=dataset
SEED_BASE=20260707

CALIBRATE_ARM_NAMES=(shared_g2 drop_ref_g0p5 drop_ref_g1 drop_ref_g1p5 drop_ref_g2)
CALIBRATE_CFG_POLICIES=(shared drop_ref drop_ref drop_ref drop_ref)
CALIBRATE_GUIDANCES=(2.0 0.5 1.0 1.5 2.0)
SHARED_SWEEP_ARM_NAMES=(shared_g0 shared_g0p25 shared_g0p5 shared_g1 shared_g2)
SHARED_SWEEP_CFG_POLICIES=(shared shared shared shared shared)
SHARED_SWEEP_GUIDANCES=(0 0.25 0.5 1.0 2.0)
BASELINE_ARM=shared_g2
ARM_NAMES=()
ARM_CFG_POLICIES=()
ARM_GUIDANCES=()
TIMING_ARGS=()

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
    echo "CFG_GUIDANCE_PROBE_FAILED rc=$rc time=$(date -u +%FT%TZ)" \
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
case "$MODE" in
  calibrate)
    [[ $EXPECTED_COUNT == 100 ]] || \
      die "MODE=calibrate requires EXPECTED_COUNT=100, got $EXPECTED_COUNT"
    [[ -z $PROMOTED_GUIDANCE ]] || \
      die "MODE=calibrate rejects PROMOTED_GUIDANCE; got $PROMOTED_GUIDANCE"
    ARM_NAMES=("${CALIBRATE_ARM_NAMES[@]}")
    ARM_CFG_POLICIES=("${CALIBRATE_CFG_POLICIES[@]}")
    ARM_GUIDANCES=("${CALIBRATE_GUIDANCES[@]}")
    expected_arm_count=5
    ;;
  promote)
    [[ $EXPECTED_COUNT == 300 ]] || \
      die "MODE=promote requires EXPECTED_COUNT=300, got $EXPECTED_COUNT"
    case "$PROMOTED_GUIDANCE" in
      0.5) promoted_arm=drop_ref_g0p5 ;;
      1.0) promoted_arm=drop_ref_g1 ;;
      1.5) promoted_arm=drop_ref_g1p5 ;;
      2.0) promoted_arm=drop_ref_g2 ;;
      *)
        die "MODE=promote requires PROMOTED_GUIDANCE in {0.5,1.0,1.5,2.0}; got ${PROMOTED_GUIDANCE:-unset}"
        ;;
    esac
    ARM_NAMES=(shared_g2 "$promoted_arm")
    ARM_CFG_POLICIES=(shared drop_ref)
    ARM_GUIDANCES=(2.0 "$PROMOTED_GUIDANCE")
    expected_arm_count=2
    ;;
  shared_sweep)
    case "$EXPECTED_COUNT" in
      5|100) ;;
      *)
        die "MODE=shared_sweep requires EXPECTED_COUNT in {5,100}, got $EXPECTED_COUNT"
        ;;
    esac
    [[ -z $PROMOTED_GUIDANCE ]] || \
      die "MODE=shared_sweep rejects PROMOTED_GUIDANCE; got $PROMOTED_GUIDANCE"
    [[ $STEPS_PER_BLOCK == 16 ]] || \
      die "MODE=shared_sweep requires STEPS_PER_BLOCK=16, got $STEPS_PER_BLOCK"
    ARM_NAMES=("${SHARED_SWEEP_ARM_NAMES[@]}")
    ARM_CFG_POLICIES=("${SHARED_SWEEP_CFG_POLICIES[@]}")
    ARM_GUIDANCES=("${SHARED_SWEEP_GUIDANCES[@]}")
    TIMING_ARGS+=(--measure-token-decode)
    expected_arm_count=5
    ;;
  *) die "MODE must be calibrate, promote, or shared_sweep, got $MODE" ;;
esac
[[ ${#ARM_NAMES[@]} -eq $expected_arm_count ]] || \
  die "MODE=$MODE expected $expected_arm_count arms, got ${#ARM_NAMES[@]}"
[[ ${#ARM_CFG_POLICIES[@]} -eq ${#ARM_NAMES[@]} ]] || \
  die "CFG policy matrix length mismatch"
[[ ${#ARM_GUIDANCES[@]} -eq ${#ARM_NAMES[@]} ]] || \
  die "guidance matrix length mismatch"
for ((i = 0; i < ${#ARM_NAMES[@]}; i++)); do
  signature="${ARM_CFG_POLICIES[$i]}|${ARM_GUIDANCES[$i]}"
  for ((j = 0; j < i; j++)); do
    prior_signature="${ARM_CFG_POLICIES[$j]}|${ARM_GUIDANCES[$j]}"
    [[ $signature != "$prior_signature" ]] || \
      die "duplicate arm argv contract: ${ARM_NAMES[$j]} and ${ARM_NAMES[$i]}"
  done
done

IFS=, read -r -a GPU_ARRAY <<< "$GPU_IDS"
for ((i = 0; i < ${#GPU_ARRAY[@]}; i++)); do
  [[ ${GPU_ARRAY[$i]} =~ ^[0-9]+$ ]] || die "invalid GPU ID: ${GPU_ARRAY[$i]}"
  for ((j = 0; j < i; j++)); do
    [[ ${GPU_ARRAY[$i]} != "${GPU_ARRAY[$j]}" ]] || \
      die "duplicate GPU ID: ${GPU_ARRAY[$i]}"
  done
done
if [[ $MODE == shared_sweep ]]; then
  ((${#GPU_ARRAY[@]} >= 2)) || \
    die "MODE=shared_sweep requires at least two GPUs"
  ((${#GPU_ARRAY[@]} < EXPECTED_COUNT)) || \
    die "MODE=shared_sweep requires fewer GPU shards than EXPECTED_COUNT so post-warmup timing rows remain"
else
  ((${#GPU_ARRAY[@]} == 3)) || \
    die "MODE=$MODE requires exactly three GPUs"
fi

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

# RESULT_ROOT may be shared by many jobs; the run directory must be new. A
# repeated RUN_ID fails immediately instead of resuming or mixing artifacts.
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

# Executed code must be committed and tracked. Unrelated untracked GPFS
# artifacts are recorded in provenance but do not change the code contract.
git diff --quiet --no-ext-diff || die "tracked unstaged changes make provenance ambiguous"
git diff --cached --quiet --no-ext-diff || die "staged changes make provenance ambiguous"
for tracked in \
  training_contract_cfg_guidance_probe.sh \
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
v "CFG_GUIDANCE_PROBE_START run_id=$RUN_ID mode=$MODE commit=$commit time=$(date -u +%FT%TZ)"
v "MATRIX arms=${ARM_NAMES[*]} langs=zh,en count=$EXPECTED_COUNT checkpoint=$ck_real gpu_ids=$GPU_IDS"
v "FIXED prompt=$PROMPT_CONTRACT lang_policy=$LANG_POLICY steps=$STEPS_PER_BLOCK block_size=$BLOCK_SIZE max_blocks=$MAX_BLOCKS dtype=bf16 eos_cfg_calibration=legacy silence_stop=0 seed_base=$SEED_BASE"

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
  echo "run=cfg_drop_ref_guidance_probe"
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
  echo "mode=$MODE"
  echo "promoted_guidance=${PROMOTED_GUIDANCE:-unset}"
  echo "expected_count=$EXPECTED_COUNT"
  echo "gpu_ids=$GPU_IDS"
  echo "generation_shards=${#GPU_ARRAY[@]}"
  echo "seed_contract=torch.manual_seed($SEED_BASE + global_subset_row_index); identical ordered subset and shard count for every arm"
  echo "fixed_contract=prompt=$PROMPT_CONTRACT lang_policy=$LANG_POLICY steps=$STEPS_PER_BLOCK block_size=$BLOCK_SIZE max_blocks=$MAX_BLOCKS dtype=bf16 eos_cfg_calibration=legacy silence_stop=0"
  echo "token_decode_timing=$([[ ${#TIMING_ARGS[@]} -gt 0 ]] && echo enabled || echo disabled)"
  echo "timing_scope=core token decode only; excludes model loading, reference encoding, codec decoding, and WAV I/O"
  echo "first_packet_latency=not measured"
  echo "arms_begin"
  for ((i = 0; i < ${#ARM_NAMES[@]}; i++)); do
    echo "${ARM_NAMES[$i]} cfg_seed=${ARM_CFG_POLICIES[$i]} guidance=${ARM_GUIDANCES[$i]}"
  done
  echo "arms_end"
  echo "code_sha256_begin"
  sha256sum \
    training_contract_cfg_guidance_probe.sh \
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
    v "ARM_START arm=$arm lang=$lang lang_policy=$LANG_POLICY prompt=$PROMPT_CONTRACT cfg_seed=$cfg_policy guidance=$guidance time=$(date -u +%FT%TZ)"

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
        --eos-cfg-calibration legacy \
        --dtype bf16 \
        --prompt-contract "$PROMPT_CONTRACT" \
        --cfg-unconditional-seed-policy "$cfg_policy" \
        --item-error-policy "$ITEM_ERROR_POLICY" \
        --silence-stop-seconds 0 \
        --silence-match-codebooks "$SILENCE_MATCH_CODEBOOKS" \
        --shard "$shard/${#GPU_ARRAY[@]}" \
        "${TIMING_ARGS[@]}" \
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
      --lang-policy "$LANG_POLICY" \
      --prompt-contract "$PROMPT_CONTRACT" \
      --cfg-unconditional-seed-policy "$cfg_policy" \
      --guidance-scale "$guidance" \
      --seed-base "$SEED_BASE" \
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
v "CFG_GUIDANCE_PROBE_DONE run_id=$RUN_ID mode=$MODE result=$RES time=$(date -u +%FT%TZ)"
