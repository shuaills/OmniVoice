#!/usr/bin/env bash
# Full Seed-TTS verdict for the 1.0s / first-two-codebook force-stop rule.
# One self-terminating 3x4090 job; fresh output directories only.
set -euo pipefail

C=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/OmniVoice
MAIN=/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice
E=/opt/gpfs/users/shuai/work/block-emilia-parity/OmniVoice
L=/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice
DL=/opt/gpfs/users/yinfeng/work/OmniVoice
BASE=/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block
MODELS=$DL/download/tts_eval_models
CK=$L/exp/blockcausal_splitloss_emilia_300k_lx20/checkpoint-300000
RES=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/results/fullset_1s_cb2_20260716
SHIM=$RES/shim
VF=$RES/VERDICT.txt

cd "$C"
source "$DL/.venv/bin/activate"
export PYTHONPATH="$C"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RES/logs" "$SHIM"
: > "$VF"
v() { echo "$1" | tee -a "$VF"; }
v "FORCE_STOP_FULLSET_START commit=$(git rev-parse HEAD) time=$(date -u +%FT%TZ)"

cp "$MAIN/results_scratch_eosdecouple_50k/shim_ckpt/config.json" "$SHIM/"
for file in model.safetensors tokenizer.json tokenizer_config.json train_config.json; do
  ln -sfn "$CK/$file" "$SHIM/$file"
done

for lang in zh en; do
  TSV=$DL/download/tts_eval_datasets/seedtts_testset/$lang/test.tsv
  JSONL=$E/results_endpoint/test_$lang.jsonl
  OUT=$RES/${lang}_wavs
  mkdir -p "$OUT"
  pids=()
  for gpu in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$gpu python tests/seedtts_blockwise_gen.py \
      --tsv "$TSV" --ckpt "$SHIM" --base "$BASE" --out "$OUT" \
      --steps-per-block 16 --guidance-scale 2.0 --dtype bf16 \
      --silence-stop-seconds 1.0 --silence-match-codebooks 2 \
      --shard "$gpu/3" \
      > "$RES/logs/gen_${lang}_shard${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  [[ $rc -eq 0 ]] || { v "GEN_${lang}=FAILED"; exit 1; }
  expected=$(wc -l < "$TSV")
  actual=$(find "$OUT" -maxdepth 1 -name "*.wav" | wc -l)
  [[ $actual -eq $expected ]] || {
    v "GEN_${lang}=COUNT_MISMATCH expected=$expected actual=$actual"
    exit 1
  }
  stops=$(python3 - "$OUT" <<'PY'
import glob
import json
import os
import sys

rows = []
for path in glob.glob(os.path.join(sys.argv[1], "gen_meta_shard*.jsonl")):
    rows.extend(json.loads(line) for line in open(path))
print(sum(bool(row.get("silence_stop")) for row in rows))
PY
)
  v "GEN_${lang}=OK wavs=$actual silence_stops=$stops"

  CUDA_VISIBLE_DEVICES=0 python omnivoice/eval/wer/seedtts.py \
    --wav-path "$OUT" --test-list "$JSONL" --model-dir "$MODELS" --lang "$lang" \
    --decode-path "$RES/wer_${lang}.tsv" --batch-size 4 \
    > "$RES/logs/wer_${lang}.log" 2>&1 &
  wer_pid=$!
  CUDA_VISIBLE_DEVICES=1 python omnivoice/eval/speaker_similarity/sim.py \
    --wav-path "$OUT" --test-list "$JSONL" --model-dir "$MODELS" \
    --decode-path "$RES/sim_${lang}.tsv" \
    > "$RES/logs/sim_${lang}.log" 2>&1 &
  sim_pid=$!
  wait "$wer_pid"
  wait "$sim_pid"

  wer=$(grep -oE "Seed-TTS WER \(Avg of WERs\): [0-9.]+%" \
    "$RES/logs/wer_${lang}.log" | tail -1)
  sim=$(grep -oE "SIM-o score: [0-9.]+" "$RES/logs/sim_${lang}.log" | tail -1)
  runaway=$(python3 - "$RES/wer_${lang}.tsv" <<'PY'
import csv
import sys

rows = list(csv.reader(open(sys.argv[1]), delimiter="\t"))[1:]
print(sum(1 for row in rows if len(row) >= 2 and float(row[1]) > 0.5))
PY
)
  v "QUALITY_${lang} ${wer:-WER_FAILED} runaway=$runaway ${sim:-SIM_FAILED}"

  python3 - "$OUT" "$E/results_endpoint/r1_$lang" "$lang" <<'PY' | tee -a "$VF"
import contextlib
import glob
import os
import statistics
import sys
import wave


def duration(path):
    with contextlib.closing(wave.open(path)) as handle:
        return handle.getnframes() / handle.getframerate()


ours = {os.path.basename(path): duration(path) for path in glob.glob(sys.argv[1] + "/*.wav")}
anchor = {os.path.basename(path): duration(path) for path in glob.glob(sys.argv[2] + "/*.wav")}
ratios = sorted(ours[key] / anchor[key] for key in ours if key in anchor)
n = len(ratios)
print(
    f"DURATION_{sys.argv[3]} n={n} mean={statistics.mean(ratios):.3f} "
    f"median={statistics.median(ratios):.3f} "
    f"p95={ratios[min(n - 1, int(n * 0.95))]:.3f} "
    f"gt2={sum(value > 2 for value in ratios)} "
    f"lt0.6={sum(value < 0.6 for value in ratios)}"
)
PY
done

v "FORCE_STOP_FULLSET_DONE $(date -u +%FT%TZ)"
