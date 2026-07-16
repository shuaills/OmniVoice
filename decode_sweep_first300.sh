#!/usr/bin/env bash
# Self-terminating OMS workload for the paired first-300 decode sweep.
# It submits no child jobs, uses fresh arm directories, and fixes silence-stop off.
set -Eeuo pipefail

C=${C:-/opt/gpfs/users/shuai/work/silence-force-stop-campaign/OmniVoice}
MAIN=${MAIN:-/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice}
E=${E:-/opt/gpfs/users/shuai/work/block-emilia-parity/OmniVoice}
L=${L:-/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice}
DL=${DL:-/opt/gpfs/users/yinfeng/work/OmniVoice}
BASE=${BASE:-/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block}
MODELS=${MODELS:-$DL/download/tts_eval_models}
CK=${CK:-$L/exp/blockcausal_splitloss_emilia_300k_lx20/checkpoint-300000}
TREND=${TREND:-$MAIN/results_wer_trend}
RESULT_ROOT=${RESULT_ROOT:-/opt/gpfs/users/shuai/work/silence-force-stop-campaign/results/decode_sweep_first300}
EXPECTED_COUNT=${EXPECTED_COUNT:-300}
GPU_IDS=${GPU_IDS:-0,1,2}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${OMS_JOB_ID:-${HOSTNAME:-host}-$$}}

if [[ ! $RUN_ID =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid RUN_ID: $RUN_ID" >&2
  exit 2
fi

RES=$RESULT_ROOT/$RUN_ID
SHIM=$RES/shim
VERDICT=$RES/VERDICT.txt
REPORTER=$C/scripts/decode_sweep_report.py
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
    echo "DECODE_SWEEP_FAILED rc=$rc time=$(date -u +%FT%TZ)" | tee -a "$VERDICT" >&2
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

tsv_for_lang() {
  case "$1" in
    zh) echo "$TREND/first300.tsv" ;;
    en) echo "$E/results_autoeval/en_first300.tsv" ;;
    *) die "unsupported language: $1" ;;
  esac
}

jsonl_for_lang() {
  case "$1" in
    zh) echo "$TREND/first300.jsonl" ;;
    en) echo "$E/results_autoeval/en_first300.jsonl" ;;
    *) die "unsupported language: $1" ;;
  esac
}

anchor_for_lang() {
  echo "$E/results_endpoint/r1_$1"
}

require_dir "$C"
require_dir "$MAIN"
require_dir "$E"
require_dir "$L"
require_dir "$DL"
require_dir "$BASE"
require_dir "$MODELS"
require_dir "$CK"
require_file "$DL/.venv/bin/activate"
require_file "$REPORTER"
require_file "$C/tests/seedtts_blockwise_gen.py"
require_file "$C/omnivoice/eval/wer/seedtts.py"
require_file "$C/omnivoice/eval/speaker_similarity/sim.py"
require_file "$MAIN/results_scratch_eosdecouple_50k/shim_ckpt/config.json"
for file in model.safetensors tokenizer.json tokenizer_config.json train_config.json; do
  require_file "$CK/$file"
done

IFS=, read -r -a GPU_ARRAY <<< "$GPU_IDS"
((${#GPU_ARRAY[@]} >= 2)) || die "GPU_IDS must provide at least two GPUs"
for ((i = 0; i < ${#GPU_ARRAY[@]}; i++)); do
  [[ ${GPU_ARRAY[$i]} =~ ^[0-9]+$ ]] || die "invalid GPU ID: ${GPU_ARRAY[$i]}"
  for ((j = 0; j < i; j++)); do
    [[ ${GPU_ARRAY[$i]} != "${GPU_ARRAY[$j]}" ]] || die "duplicate GPU ID: ${GPU_ARRAY[$i]}"
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

commit=$(git rev-parse HEAD)
v "DECODE_SWEEP_START run_id=$RUN_ID commit=$commit checkpoint=$(realpath "$CK") time=$(date -u +%FT%TZ)"
v "MATRIX guidance=1.5,2.0,3.0 steps=16,32 langs=zh,en count=$EXPECTED_COUNT dtype=bf16 silence_stop_seconds=0 gpu_ids=$GPU_IDS"

cp "$MAIN/results_scratch_eosdecouple_50k/shim_ckpt/config.json" "$SHIM/config.json"
for file in model.safetensors tokenizer.json tokenizer_config.json train_config.json; do
  ln -s "$CK/$file" "$SHIM/$file"
done

for lang in zh en; do
  tsv=$(tsv_for_lang "$lang")
  jsonl=$(jsonl_for_lang "$lang")
  anchor=$(anchor_for_lang "$lang")
  require_file "$tsv"
  require_file "$jsonl"
  require_dir "$anchor"
  python "$REPORTER" validate-inputs \
    --tsv "$tsv" \
    --jsonl "$jsonl" \
    --expected-count "$EXPECTED_COUNT" \
    --ids-out "$RES/inputs/${lang}_ids.txt" \
    --manifest-out "$RES/inputs/${lang}_manifest.json" \
    --check-ref-audio | tee -a "$VERDICT"
done

cat > "$RES/run_manifest.json" <<EOF
{
  "run_id": "$RUN_ID",
  "commit": "$commit",
  "checkpoint": "$(realpath "$CK")",
  "base_model": "$(realpath "$BASE")",
  "dtype": "bf16",
  "silence_stop_seconds": 0.0,
  "guidance_scales": [1.5, 2.0, 3.0],
  "steps_per_block": [16, 32],
  "languages": ["zh", "en"],
  "expected_count_per_language": $EXPECTED_COUNT,
  "gpu_ids": "$GPU_IDS",
  "generation_shards": ${#GPU_ARRAY[@]},
  "seed_contract": "tests/seedtts_blockwise_gen.py fixed seed; identical shard count and ordered TSV for every arm"
}
EOF

# Baseline first, then the remaining five cells. Every cell is regenerated.
GUIDANCES=(2.0 1.5 3.0 2.0 1.5 3.0)
STEPS=(16 16 16 32 32 32)
for ((arm_index = 0; arm_index < ${#GUIDANCES[@]}; arm_index++)); do
  guidance=${GUIDANCES[$arm_index]}
  steps=${STEPS[$arm_index]}
  arm="g${guidance//./p}_s${steps}"
  for lang in zh en; do
    tsv=$(tsv_for_lang "$lang")
    jsonl=$(jsonl_for_lang "$lang")
    anchor=$(anchor_for_lang "$lang")
    arm_dir=$RES/arms/$arm/$lang
    wav_dir=$arm_dir/wavs
    mkdir -p "$arm_dir/logs"
    mkdir "$wav_dir"
    v "ARM_START arm=$arm lang=$lang guidance=$guidance steps=$steps time=$(date -u +%FT%TZ)"

    PIDS=()
    for ((shard = 0; shard < ${#GPU_ARRAY[@]}; shard++)); do
      gpu=${GPU_ARRAY[$shard]}
      CUDA_VISIBLE_DEVICES=$gpu python tests/seedtts_blockwise_gen.py \
        --tsv "$tsv" \
        --ckpt "$SHIM" \
        --base "$BASE" \
        --out "$wav_dir" \
        --steps-per-block "$steps" \
        --guidance-scale "$guidance" \
        --dtype bf16 \
        --lang "$lang" \
        --silence-stop-seconds 0 \
        --silence-match-codebooks 2 \
        --shard "$shard/${#GPU_ARRAY[@]}" \
        > "$arm_dir/logs/gen_shard${shard}.log" 2>&1 &
      PIDS+=("$!")
    done
    wait_group "generation arm=$arm lang=$lang"

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
      --guidance-scale "$guidance" \
      --steps-per-block "$steps" \
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
      --summary-out "$arm_dir/summary.json" | tee -a "$VERDICT"
  done
done

shopt -s nullglob
SUMMARY_FILES=("$RES"/arms/*/*/summary.json)
shopt -u nullglob
((${#SUMMARY_FILES[@]} == 12)) || die "expected 12 arm/language summaries, got ${#SUMMARY_FILES[@]}"
python "$REPORTER" aggregate \
  --summaries "${SUMMARY_FILES[@]}" \
  --expected-languages zh,en \
  --expected-guidance 1.5,2.0,3.0 \
  --expected-steps 16,32 \
  --expected-count "$EXPECTED_COUNT" \
  --output-tsv "$RES/SUMMARY.tsv" \
  --output-json "$RES/SUMMARY.json" \
  --output-md "$RES/SUMMARY.md" | tee -a "$VERDICT"

tee -a "$VERDICT" < "$RES/SUMMARY.md"
v "DECODE_SWEEP_DONE run_id=$RUN_ID result=$RES time=$(date -u +%FT%TZ)"
