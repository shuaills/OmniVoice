#!/usr/bin/env bash
# Exact en300 control-vs-force-stop check. Self-terminates after scoring.
set -euo pipefail

C=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/OmniVoice
MAIN=/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice
E=/opt/gpfs/users/shuai/work/block-emilia-parity/OmniVoice
DL=/opt/gpfs/users/yinfeng/work/OmniVoice
BASE=/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block
MODELS=$DL/download/tts_eval_models
RES=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/results/pilot_en300_20260716
SHIM=$RES/shim
TSV=$E/results_autoeval/en_first300.tsv
JSONL=$E/results_autoeval/en_first300.jsonl
CONTROL=$RES/baseline_wavs
FORCED=$RES/forced_1s_cb2_wavs
VF=$RES/paired_verdict.txt

cd "$C"
source "$DL/.venv/bin/activate"
export PYTHONPATH="$C"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RES/logs" "$FORCED"
: > "$VF"
v() { echo "$1" | tee -a "$VF"; }

pids=()
for gpu in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$gpu python tests/seedtts_blockwise_gen.py \
    --tsv "$TSV" --ckpt "$SHIM" --base "$BASE" --out "$FORCED" \
    --steps-per-block 16 --guidance-scale 2.0 --dtype bf16 \
    --silence-stop-seconds 1.0 --silence-match-codebooks 2 \
    --shard "$gpu/3" \
    > "$RES/logs/forced_gen_shard${gpu}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
if [[ $rc -ne 0 ]]; then
  v "FORCED_GENERATION_FAILED"
  exit 1
fi

for arm in control forced; do
  wavs=$CONTROL
  [[ $arm == forced ]] && wavs=$FORCED
  CUDA_VISIBLE_DEVICES=0 python omnivoice/eval/wer/seedtts.py \
    --wav-path "$wavs" --test-list "$JSONL" --model-dir "$MODELS" --lang en \
    --decode-path "$RES/wer_${arm}.tsv" --batch-size 4 \
    > "$RES/logs/wer_${arm}.log" 2>&1
  wer=$(grep -oE "Seed-TTS WER \(Avg of WERs\): [0-9.]+%" \
    "$RES/logs/wer_${arm}.log" | tail -1)
  runaway=$(python3 - "$RES/wer_${arm}.tsv" <<'PY'
import csv
import sys

rows = list(csv.reader(open(sys.argv[1]), delimiter="\t"))[1:]
print(sum(1 for row in rows if len(row) >= 2 and float(row[1]) > 0.5))
PY
)
  v "WER_${arm}=${wer:-FAILED} runaway=$runaway"
done

python3 - "$CONTROL" "$FORCED" "$E/results_endpoint/r1_en" <<'PY' | tee -a "$VF"
import contextlib
import glob
import os
import statistics
import sys
import wave


def duration(path):
    with contextlib.closing(wave.open(path)) as handle:
        return handle.getnframes() / handle.getframerate()


def collect(path):
    return {os.path.basename(item): duration(item) for item in glob.glob(path + "/*.wav")}


control, forced, anchor = map(collect, sys.argv[1:])
for name, values in (("control", control), ("forced", forced)):
    ratios = sorted(values[key] / anchor[key] for key in values if key in anchor)
    n = len(ratios)
    print(
        f"DUR_{name} n={n} mean={statistics.mean(ratios):.3f} "
        f"median={statistics.median(ratios):.3f} "
        f"p95={ratios[min(n - 1, int(n * 0.95))]:.3f} "
        f"gt2={sum(value > 2 for value in ratios)} "
        f"lt0.6={sum(value < 0.6 for value in ratios)}"
    )
changed = [key for key in forced if key in control and forced[key] < control[key]]
print(f"FORCE_STOP_COUNT={len(changed)}")
PY

v "SILENCE_FORCE_STOP_PAIRED_PILOT_DONE $(date -u +%FT%TZ)"
