#!/usr/bin/env bash
# Paired full Seed-TTS verdict for the 1.0s / first-two-codebook force-stop rule.
# One self-terminating 3x4090 workload: control + forced generation and scoring.
set -euo pipefail

C=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/OmniVoice
MAIN=/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice
E=/opt/gpfs/users/shuai/work/block-emilia-parity/OmniVoice
L=/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice
DL=/opt/gpfs/users/yinfeng/work/OmniVoice
BASE=/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block
MODELS=$DL/download/tts_eval_models
CK=$L/exp/blockcausal_splitloss_emilia_300k_lx20/checkpoint-300000
CONFIG_SRC=$MAIN/results_scratch_eosdecouple_50k/shim_ckpt/config.json
RES=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/results/fullset_paired_1s_cb2_20260716_v2
SHIM=$RES/shim
VF=$RES/VERDICT.txt
PROV=$RES/PROVENANCE.txt
NUM_SHARDS=3
MATCH_CODEBOOKS=2
SILENCE_SECONDS=1.0

die() {
  echo "FATAL: $*" >&2
  exit 1
}

# The generator skips existing wavs.  A verdict run therefore accepts only an
# absent or literally empty result directory; it never resumes or overwrites.
mkdir -p "$(dirname "$RES")"
if [[ -e "$RES" ]]; then
  [[ -d "$RES" ]] || die "result path exists and is not a directory: $RES"
  [[ -z "$(find "$RES" -mindepth 1 -print -quit)" ]] || \
    die "result directory is not empty; choose a new tag: $RES"
else
  mkdir "$RES"
fi
mkdir "$RES/logs" "$SHIM"
: > "$VF"
v() { echo "$1" | tee -a "$VF"; }

ACTIVE_PIDS=()
cleanup_children() {
  local status=$?
  trap - EXIT
  for pid in "${ACTIVE_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$C"
[[ -x "$DL/.venv/bin/python" ]] || die "venv missing: $DL/.venv"
for path in \
  "$CK/model.safetensors" \
  "$CONFIG_SRC" \
  "$BASE" \
  "$MODELS" \
  tests/seedtts_blockwise_gen.py \
  tools/validate_silence_force_stop_run.py; do
  [[ -e "$path" ]] || die "required input missing: $path"
done
for lang in zh en; do
  [[ -f "$DL/download/tts_eval_datasets/seedtts_testset/$lang/test.tsv" ]] || \
    die "dataset TSV missing for $lang"
  [[ -f "$E/results_endpoint/test_$lang.jsonl" ]] || \
    die "score JSONL missing for $lang"
  [[ -d "$E/results_endpoint/r1_$lang" ]] || \
    die "duration anchor missing for $lang"
done

git diff --quiet --no-ext-diff || \
  die "repository has tracked unstaged changes; provenance would be ambiguous"
git diff --cached --quiet --no-ext-diff || \
  die "repository has tracked staged changes; provenance would be ambiguous"
commit=$(git rev-parse HEAD)
ck_real=$(readlink -f "$CK")
config_real=$(readlink -f "$CONFIG_SRC")
base_real=$(readlink -f "$BASE")

source "$DL/.venv/bin/activate"
export PYTHONPATH="$C"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cp "$CONFIG_SRC" "$SHIM/config.json"
ln -s "$ck_real/model.safetensors" "$SHIM/model.safetensors"

{
  echo "run=paired_seedtts_silence_force_stop_fullset"
  echo "created_utc=$(date -u +%FT%TZ)"
  echo "repo=$C"
  echo "commit=$commit"
  echo "git_tracked_status=clean"
  echo "git_untracked_paths_begin"
  git ls-files --others --exclude-standard | LC_ALL=C sort
  echo "git_untracked_paths_end"
  echo "checkpoint_requested=$CK"
  echo "checkpoint_resolved=$ck_real"
  echo "shim_config_source=$config_real"
  echo "base_model_resolved=$base_real"
  echo "generator=tests/seedtts_blockwise_gen.py"
  echo "seed_contract=torch.manual_seed(20260707 + dataset_row_index)"
  echo "paired_contract=same commit, checkpoint, base, TSV order, shard, seed, bf16, steps=16, guidance=2.0"
  echo "control_detector_seconds=0"
  echo "forced_detector_seconds=$SILENCE_SECONDS"
  echo "silence_match_codebooks=$MATCH_CODEBOOKS"
  echo "num_shards=$NUM_SHARDS"
  echo "code_sha256:"
  sha256sum \
    silence_force_stop_fullset.sh \
    tests/seedtts_blockwise_gen.py \
    tools/validate_silence_force_stop_run.py \
    omnivoice/blockdiff_dual.py
  echo "checkpoint_sha256:"
  sha256sum "$ck_real/model.safetensors" "$config_real"
  echo "input_sha256:"
  for lang in zh en; do
    sha256sum \
      "$DL/download/tts_eval_datasets/seedtts_testset/$lang/test.tsv" \
      "$E/results_endpoint/test_$lang.jsonl"
  done
} > "$PROV"

v "FORCE_STOP_FULLSET_START commit=$commit time=$(date -u +%FT%TZ)"
v "PROVENANCE=$PROV checkpoint=$ck_real dtype=bf16 paired=true"

generate_arm() {
  local lang=$1
  local arm=$2
  local stop_seconds=$3
  local tsv=$4
  local out=$5
  local rc=0

  mkdir "$out"
  ACTIVE_PIDS=()
  for gpu in $(seq 0 $((NUM_SHARDS - 1))); do
    # Materialize every shard ledger even when it stays empty.  The audit
    # requires exactly one metadata file and one empty failure file per shard.
    : > "$out/gen_meta_shard${gpu}.jsonl"
    : > "$out/failures_shard${gpu}.jsonl"
    CUDA_VISIBLE_DEVICES=$gpu python tests/seedtts_blockwise_gen.py \
      --tsv "$tsv" --ckpt "$SHIM" --base "$BASE" --out "$out" \
      --steps-per-block 16 --guidance-scale 2.0 --dtype bf16 \
      --silence-stop-seconds "$stop_seconds" \
      --silence-match-codebooks "$MATCH_CODEBOOKS" \
      --shard "$gpu/$NUM_SHARDS" \
      > "$RES/logs/gen_${lang}_${arm}_shard${gpu}.log" 2>&1 &
    ACTIVE_PIDS+=("$!")
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  ACTIVE_PIDS=()
  [[ $rc -eq 0 ]] || die "generation process failed: lang=$lang arm=$arm"
}

audit_generation() {
  local lang=$1
  local tsv=$2
  local jsonl=$3
  local control=$4
  local forced=$5
  python tools/validate_silence_force_stop_run.py \
    --lang "$lang" \
    --dataset-tsv "$tsv" \
    --test-jsonl "$jsonl" \
    --control-dir "$control" \
    --forced-dir "$forced" \
    --logs-dir "$RES/logs" \
    --num-shards "$NUM_SHARDS" \
    --match-codebooks "$MATCH_CODEBOOKS" \
    --summary-json "$RES/audit_generation_${lang}.json" \
    | tee -a "$VF"
}

score_arm() {
  local lang=$1
  local arm=$2
  local jsonl=$3
  local wavs=$4
  local rc=0

  CUDA_VISIBLE_DEVICES=0 python omnivoice/eval/wer/seedtts.py \
    --wav-path "$wavs" --test-list "$jsonl" --model-dir "$MODELS" --lang "$lang" \
    --decode-path "$RES/wer_${lang}_${arm}.tsv" --batch-size 4 \
    > "$RES/logs/wer_${lang}_${arm}.log" 2>&1 &
  local wer_pid=$!
  CUDA_VISIBLE_DEVICES=1 python omnivoice/eval/speaker_similarity/sim.py \
    --wav-path "$wavs" --test-list "$jsonl" --model-dir "$MODELS" \
    --decode-path "$RES/sim_${lang}_${arm}.tsv" \
    > "$RES/logs/sim_${lang}_${arm}.log" 2>&1 &
  local sim_pid=$!
  ACTIVE_PIDS=("$wer_pid" "$sim_pid")
  if ! wait "$wer_pid"; then
    rc=1
  fi
  if ! wait "$sim_pid"; then
    rc=1
  fi
  ACTIVE_PIDS=()
  [[ $rc -eq 0 ]] || die "scoring process failed: lang=$lang arm=$arm"

  local wer sim runaway
  wer=$(grep -oE "Seed-TTS WER \(Avg of WERs\): [0-9.]+%" \
    "$RES/logs/wer_${lang}_${arm}.log" | tail -1 || true)
  sim=$(grep -oE "SIM-o score: [0-9.]+" \
    "$RES/logs/sim_${lang}_${arm}.log" | tail -1 || true)
  [[ -n "$wer" ]] || die "WER summary parse failed: lang=$lang arm=$arm"
  [[ -n "$sim" ]] || die "SIM summary parse failed: lang=$lang arm=$arm"
  runaway=$(python - "$RES/wer_${lang}_${arm}.tsv" <<'PY'
import csv
import sys

rows = list(csv.reader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))[1:]
print(sum(1 for row in rows if len(row) >= 2 and row[0].endswith(".wav") and float(row[1]) > 0.5))
PY
)
  v "QUALITY_${lang}_${arm} $wer wer_gt50pct=$runaway $sim"
}

audit_complete() {
  local lang=$1
  local tsv=$2
  local jsonl=$3
  local control=$4
  local forced=$5
  python tools/validate_silence_force_stop_run.py \
    --lang "$lang" \
    --dataset-tsv "$tsv" \
    --test-jsonl "$jsonl" \
    --control-dir "$control" \
    --forced-dir "$forced" \
    --logs-dir "$RES/logs" \
    --num-shards "$NUM_SHARDS" \
    --match-codebooks "$MATCH_CODEBOOKS" \
    --wer-control "$RES/wer_${lang}_control.tsv" \
    --wer-forced "$RES/wer_${lang}_forced.tsv" \
    --sim-control "$RES/sim_${lang}_control.tsv" \
    --sim-forced "$RES/sim_${lang}_forced.tsv" \
    --summary-json "$RES/audit_complete_${lang}.json" \
    | tee -a "$VF"
}

duration_verdict() {
  local lang=$1
  local tsv=$2
  local control=$3
  local forced=$4
  local anchor=$5
  python - "$tsv" "$control" "$forced" "$anchor" "$lang" <<'PY' | tee -a "$VF"
import contextlib
import csv
import statistics
import sys
import wave
from pathlib import Path


def duration(path):
    with contextlib.closing(wave.open(str(path))) as handle:
        return handle.getnframes() / handle.getframerate()


def collect(path):
    return {item.stem: duration(item) for item in Path(path).glob("*.wav")}


with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    expected = {row[0] for row in csv.reader(handle, delimiter="\t") if row}
control, forced, anchor = map(collect, sys.argv[2:5])
for name, values in (("control", control), ("forced", forced), ("anchor", anchor)):
    if set(values) != expected:
        missing = sorted(expected - set(values))[:8]
        extra = sorted(set(values) - expected)[:8]
        raise SystemExit(f"DURATION_ID_MISMATCH arm={name} missing={missing} extra={extra}")
for name, values in (("control", control), ("forced", forced)):
    ratios = sorted(values[key] / anchor[key] for key in expected)
    n = len(ratios)
    print(
        f"DURATION_{sys.argv[5]}_{name} n={n} "
        f"mean={statistics.mean(ratios):.3f} "
        f"median={statistics.median(ratios):.3f} "
        f"p95={ratios[min(n - 1, int(n * 0.95))]:.3f} "
        f"gt2={sum(value > 2 for value in ratios)} "
        f"lt0.6={sum(value < 0.6 for value in ratios)}"
    )
PY
}

for lang in zh en; do
  TSV=$DL/download/tts_eval_datasets/seedtts_testset/$lang/test.tsv
  JSONL=$E/results_endpoint/test_$lang.jsonl
  CONTROL=$RES/${lang}_control_wavs
  FORCED=$RES/${lang}_forced_1s_cb2_wavs

  v "GEN_${lang}_PAIR_START time=$(date -u +%FT%TZ)"
  generate_arm "$lang" control 0 "$TSV" "$CONTROL"
  generate_arm "$lang" forced "$SILENCE_SECONDS" "$TSV" "$FORCED"
  audit_generation "$lang" "$TSV" "$JSONL" "$CONTROL" "$FORCED"

  score_arm "$lang" control "$JSONL" "$CONTROL"
  score_arm "$lang" forced "$JSONL" "$FORCED"
  audit_complete "$lang" "$TSV" "$JSONL" "$CONTROL" "$FORCED"
  duration_verdict "$lang" "$TSV" "$CONTROL" "$FORCED" "$E/results_endpoint/r1_$lang"
done

v "FORCE_STOP_FULLSET_DONE $(date -u +%FT%TZ)"
find "$RES" -type f ! -name "*.wav" ! -name "ARTIFACTS.sha256" -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$RES/ARTIFACTS.sha256"
