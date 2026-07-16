#!/usr/bin/env bash
# Self-terminating paired evaluation for the bandctl/band4 10k checkpoints.
set -Eeuo pipefail

C=${C:-/opt/gpfs/users/shuai/work/training-contract-probes/OmniVoice}
MAIN=${MAIN:-/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice}
E=${E:-/opt/gpfs/users/shuai/work/block-emilia-parity/OmniVoice}
DL=${DL:-/opt/gpfs/users/yinfeng/work/OmniVoice}
BASE=${BASE:-/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block}
MODELS=${MODELS:-$DL/download/tts_eval_models}
TREND=${TREND:-$MAIN/results_wer_trend}
BANDCTL_CK=${BANDCTL_CK:-$C/exp/splitloss_ft10k_eos_bandctl_shuai-bandctl-ft10k-4g-v2/checkpoint-10000}
BAND4_CK=${BAND4_CK:-$C/exp/splitloss_ft10k_eos_band4_shuai-band4-ft10k-4g-v2/checkpoint-10000}

EXPECTED_COUNT=${EXPECTED_COUNT:-100}
GPU_IDS=${GPU_IDS:-0,1,2}
STEPS_PER_BLOCK=16
BLOCK_SIZE=32
MAX_BLOCKS=24
GUIDANCE_SCALE=2.0
RESULT_ROOT=${RESULT_ROOT:-/opt/gpfs/users/shuai/work/training-contract-probes/results/band-ft10k-first${EXPECTED_COUNT}}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${OMS_JOB_ID:-${HOSTNAME:-host}-$$}}

ARMS=(bandctl band4)
CHECKPOINTS=("$BANDCTL_CK" "$BAND4_CK")
BASELINE_ARM=bandctl
REPORTER=$C/scripts/training_contract_probe_report.py
GENERATOR=$C/tests/seedtts_blockwise_gen.py
VOCAB_PREFLIGHT=$C/scripts/check_checkpoint_vocab.py
RES=$RESULT_ROOT/$RUN_ID
VERDICT=$RES/VERDICT.txt
PROVENANCE=$RES/PROVENANCE.txt
PIDS=()

die() { echo "ERROR: $*" >&2; return 1; }
require_file() { [[ -f $1 ]] || die "required file not found: $1"; }
require_dir() { [[ -d $1 ]] || die "required directory not found: $1"; }

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if ((${#PIDS[@]})); then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  if ((rc != 0)) && [[ -n ${VERDICT:-} && -e ${RES:-} ]]; then
    echo "BAND_FT10K_EVAL_FAILED rc=$rc time=$(date -u +%FT%TZ)" | tee -a "$VERDICT" >&2
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_group() {
  local label=$1 failed=0 pid
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

anchor_for_lang() { echo "$E/results_endpoint/r1_$1"; }

[[ $RUN_ID =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid RUN_ID: $RUN_ID"
[[ $EXPECTED_COUNT =~ ^[0-9]+$ ]] || die "EXPECTED_COUNT must be a positive integer"
((EXPECTED_COUNT >= 1 && EXPECTED_COUNT <= 300)) || die "EXPECTED_COUNT must be between 1 and 300"
IFS=, read -r -a GPU_ARRAY <<< "$GPU_IDS"
((${#GPU_ARRAY[@]} == 3)) || die "GPU_IDS must contain exactly three GPUs"
for ((i = 0; i < 3; i++)); do
  [[ ${GPU_ARRAY[$i]} =~ ^[0-9]+$ ]] || die "invalid GPU ID: ${GPU_ARRAY[$i]}"
  for ((j = 0; j < i; j++)); do
    [[ ${GPU_ARRAY[$i]} != "${GPU_ARRAY[$j]}" ]] || die "duplicate GPU ID: ${GPU_ARRAY[$i]}"
  done
done

require_dir "$C"
require_dir "$DL"
require_dir "$BASE"
require_dir "$BASE/audio_tokenizer"
require_dir "$MODELS"
require_file "$DL/.venv/bin/activate"
require_file "$REPORTER"
require_file "$GENERATOR"
require_file "$VOCAB_PREFLIGHT"
for checkpoint in "${CHECKPOINTS[@]}"; do
  require_dir "$checkpoint"
  require_file "$checkpoint/model.safetensors"
  require_file "$checkpoint/config.json"
  require_file "$checkpoint/tokenizer.json"
  require_file "$checkpoint/tokenizer_config.json"
done
for lang in zh en; do
  require_file "$(source_tsv_for_lang "$lang")"
  require_file "$(source_jsonl_for_lang "$lang")"
  require_dir "$(anchor_for_lang "$lang")"
done

mkdir -p "$RESULT_ROOT"
mkdir "$RES"
mkdir "$RES/logs" "$RES/inputs" "$RES/arms" "$RES/shims"
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
  band_ft10k_eval_pair.sh \
  scripts/training_contract_probe_report.py \
  tests/seedtts_blockwise_gen.py \
  scripts/check_checkpoint_vocab.py; do
  git ls-files --error-unmatch "$tracked" >/dev/null 2>&1 || die "executed code is not tracked: $tracked"
done

commit=$(git rev-parse HEAD)
branch=$(git branch --show-current)
v "BAND_FT10K_EVAL_START run_id=$RUN_ID commit=$commit time=$(date -u +%FT%TZ)"
v "MATRIX arms=${ARMS[*]} langs=zh,en count=$EXPECTED_COUNT gpu_ids=$GPU_IDS"
v "FIXED steps=$STEPS_PER_BLOCK block_size=$BLOCK_SIZE max_blocks=$MAX_BLOCKS guidance=$GUIDANCE_SCALE dtype=bf16 silence_stop=0"

for ((i = 0; i < ${#ARMS[@]}; i++)); do
  arm=${ARMS[$i]}
  checkpoint=${CHECKPOINTS[$i]}
  python "$VOCAB_PREFLIGHT" --checkpoint "$checkpoint" | tee -a "$VERDICT"
  shim=$RES/shims/$arm
  mkdir "$shim"
  cp "$checkpoint/config.json" "$shim/config.json"
  ln -s "$(realpath "$checkpoint/model.safetensors")" "$shim/model.safetensors"
  ln -s "$(realpath "$BASE/audio_tokenizer")" "$shim/audio_tokenizer"
  for file in tokenizer.json tokenizer_config.json train_config.json chat_template.jinja; do
    [[ ! -f "$checkpoint/$file" ]] || ln -s "$(realpath "$checkpoint/$file")" "$shim/$file"
  done
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
  echo "run=band_ft10k_eval_pair"
  echo "created_utc=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname -f 2>/dev/null || hostname)"
  echo "oms_job_id=${OMS_JOB_ID:-unset}"
  echo "repo=$(realpath "$C")"
  echo "branch=$branch"
  echo "commit=$commit"
  echo "bandctl_checkpoint=$(realpath "$BANDCTL_CK")"
  echo "band4_checkpoint=$(realpath "$BAND4_CK")"
  echo "expected_count=$EXPECTED_COUNT"
  echo "gpu_ids=$GPU_IDS"
  echo "seed_contract=torch.manual_seed(20260707 + global_subset_row_index); identical ordered subset and 3-shard layout for both checkpoints"
  echo "code_sha256_begin"
  sha256sum band_ft10k_eval_pair.sh "$REPORTER" "$GENERATOR" "$VOCAB_PREFLIGHT" \
    omnivoice/blockdiff_dual.py omnivoice/models/omnivoice.py
  echo "code_sha256_end"
  echo "checkpoint_sha256_begin"
  for checkpoint in "${CHECKPOINTS[@]}"; do
    sha256sum "$checkpoint/model.safetensors" "$checkpoint/config.json"
  done
  echo "checkpoint_sha256_end"
  echo "input_manifest_sha256_begin"
  sha256sum "$RES/inputs/zh/manifest.json" "$RES/inputs/en/manifest.json"
  echo "input_manifest_sha256_end"
  python --version
  uname -a
  nvidia-smi -L
} > "$PROVENANCE"

for ((arm_index = 0; arm_index < ${#ARMS[@]}; arm_index++)); do
  arm=${ARMS[$arm_index]}
  shim=$RES/shims/$arm
  BASELINE_ARGS=()
  [[ $arm != "$BASELINE_ARM" ]] || BASELINE_ARGS=(--is-baseline)
  for lang in zh en; do
    tsv=$RES/inputs/$lang/test.tsv
    jsonl=$RES/inputs/$lang/test.jsonl
    arm_dir=$RES/arms/$arm/$lang
    wav_dir=$arm_dir/wavs
    mkdir -p "$arm_dir/logs"
    mkdir "$wav_dir"
    v "ARM_START arm=$arm lang=$lang time=$(date -u +%FT%TZ)"

    PIDS=()
    for ((shard = 0; shard < 3; shard++)); do
      gpu=${GPU_ARRAY[$shard]}
      : > "$wav_dir/gen_meta_shard${shard}.jsonl"
      : > "$wav_dir/failures_shard${shard}.jsonl"
      CUDA_VISIBLE_DEVICES=$gpu python "$GENERATOR" \
        --tsv "$tsv" --ckpt "$shim" --base "$shim" --out "$wav_dir" \
        --steps-per-block "$STEPS_PER_BLOCK" --max-blocks "$MAX_BLOCKS" \
        --block-size "$BLOCK_SIZE" --guidance-scale "$GUIDANCE_SCALE" \
        --dtype bf16 --prompt-contract current \
        --cfg-unconditional-seed-policy shared --item-error-policy fail-at-end \
        --silence-stop-seconds 0 --silence-match-codebooks 2 \
        --shard "$shard/3" > "$arm_dir/logs/gen_shard${shard}.log" 2>&1 &
      PIDS+=("$!")
    done
    wait_group "generation arm=$arm lang=$lang"

    python "$REPORTER" validate-generation \
      --tsv "$tsv" --jsonl "$jsonl" --expected-count "$EXPECTED_COUNT" \
      --num-shards 3 --arm "$arm" --lang "$lang" --lang-policy dataset \
      --prompt-contract current --cfg-unconditional-seed-policy shared \
      --guidance-scale "$GUIDANCE_SCALE" --wav-dir "$wav_dir" \
      --output "$arm_dir/generation_audit.json" "${BASELINE_ARGS[@]}" | tee -a "$VERDICT"

    PIDS=()
    CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} python omnivoice/eval/wer/seedtts.py \
      --wav-path "$wav_dir" --test-list "$jsonl" --model-dir "$MODELS" \
      --lang "$lang" --decode-path "$arm_dir/wer.tsv" --batch-size 4 \
      > "$arm_dir/logs/wer.log" 2>&1 &
    PIDS+=("$!")
    CUDA_VISIBLE_DEVICES=${GPU_ARRAY[1]} python omnivoice/eval/speaker_similarity/sim.py \
      --wav-path "$wav_dir" --test-list "$jsonl" --model-dir "$MODELS" \
      --decode-path "$arm_dir/sim.tsv" > "$arm_dir/logs/sim.log" 2>&1 &
    PIDS+=("$!")
    wait_group "scoring arm=$arm lang=$lang"

    python "$REPORTER" summarize-arm \
      --arm "$arm" --lang "$lang" --lang-policy dataset \
      --prompt-contract current --cfg-unconditional-seed-policy shared "${BASELINE_ARGS[@]}" \
      --guidance-scale "$GUIDANCE_SCALE" --steps-per-block "$STEPS_PER_BLOCK" \
      --expected-count "$EXPECTED_COUNT" --tsv "$tsv" --jsonl "$jsonl" \
      --wav-dir "$wav_dir" --anchor-dir "$(anchor_for_lang "$lang")" \
      --wer-tsv "$arm_dir/wer.tsv" --sim-tsv "$arm_dir/sim.tsv" \
      --meta-glob "$wav_dir/gen_meta_shard*.jsonl" \
      --failures-glob "$wav_dir/failures_shard*.jsonl" \
      --per-utt-out "$arm_dir/per_utt.tsv" --summary-out "$arm_dir/summary.json" \
      | tee -a "$VERDICT"

    python - "$arm_dir/per_utt.tsv" "$arm_dir/frame_mod32.json" "$arm" "$lang" <<'PY' | tee -a "$VERDICT"
import csv
import json
import sys
from collections import Counter

source, output, arm, lang = sys.argv[1:]
with open(source, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
remainders = Counter(int(row["frames"]) % 32 for row in rows)
eos_partial = sum(int(row["eos"]) == 1 and int(row["frames"]) % 32 != 0 for row in rows)
payload = {
    "arm": arm,
    "lang": lang,
    "count": len(rows),
    "block_size": 32,
    "eos_partial_block_count": eos_partial,
    "frames_mod32_histogram": {str(key): remainders.get(key, 0) for key in range(32)},
}
with open(output, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"FRAME_MOD32_OK arm={arm} lang={lang} eos_partial={eos_partial}/{len(rows)} histogram={dict(sorted(remainders.items()))}")
PY
  done
done

shopt -s nullglob
SUMMARY_FILES=("$RES"/arms/*/*/summary.json)
shopt -u nullglob
((${#SUMMARY_FILES[@]} == 4)) || die "expected 4 summaries, got ${#SUMMARY_FILES[@]}"
python "$REPORTER" aggregate \
  --summaries "${SUMMARY_FILES[@]}" --expected-arms bandctl,band4 \
  --expected-languages zh,en --baseline-arm "$BASELINE_ARM" \
  --expected-count "$EXPECTED_COUNT" --output-tsv "$RES/SUMMARY.tsv" \
  --output-json "$RES/SUMMARY.json" --output-md "$RES/SUMMARY.md" | tee -a "$VERDICT"
tee -a "$VERDICT" < "$RES/SUMMARY.md"
v "BAND_FT10K_EVAL_DONE run_id=$RUN_ID result=$RES time=$(date -u +%FT%TZ)"
