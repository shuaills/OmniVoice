#!/usr/bin/env bash
# Score the existing en300 control/force-stop pair for speaker similarity.
# This is a self-terminating, scoring-only workload; it never regenerates audio.
set -euo pipefail

C=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/OmniVoice
E=/opt/gpfs/users/shuai/work/block-emilia-parity/OmniVoice
DL=/opt/gpfs/users/yinfeng/work/OmniVoice
MODELS=$DL/download/tts_eval_models
PILOT=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/results/pilot_en300_20260716
JSONL=$E/results_autoeval/en_first300.jsonl
OUT=$PILOT/paired_sim_v1
VERDICT=$OUT/VERDICT.txt

[[ ! -e "$OUT" ]] || {
  echo "refusing to reuse result directory: $OUT" >&2
  exit 2
}
for path in "$PILOT/baseline_wavs" "$PILOT/forced_1s_cb2_wavs" "$JSONL"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done

mkdir -p "$OUT/logs"
cd "$C"
source "$DL/.venv/bin/activate"
export PYTHONPATH=$C

{
  echo "PAIRED_SIM_START commit=$(git rev-parse HEAD) time=$(date -u +%FT%TZ)"
  echo "dataset_sha256=$(sha256sum "$JSONL" | awk '{print $1}')"
  echo "control_count=$(find "$PILOT/baseline_wavs" -maxdepth 1 -name '*.wav' | wc -l)"
  echo "forced_count=$(find "$PILOT/forced_1s_cb2_wavs" -maxdepth 1 -name '*.wav' | wc -l)"
} | tee "$VERDICT"

for arm in control forced; do
  wavs=$PILOT/baseline_wavs
  [[ $arm == forced ]] && wavs=$PILOT/forced_1s_cb2_wavs
  CUDA_VISIBLE_DEVICES=0 python omnivoice/eval/speaker_similarity/sim.py \
    --wav-path "$wavs" --test-list "$JSONL" --model-dir "$MODELS" \
    --decode-path "$OUT/sim_${arm}.tsv" \
    > "$OUT/logs/sim_${arm}.log" 2>&1
  score=$(grep -oE 'SIM-o score: [0-9.]+' "$OUT/logs/sim_${arm}.log" | tail -1)
  [[ -n $score ]] || { echo "SIM_${arm}=FAILED" | tee -a "$VERDICT"; exit 1; }
  rows=$(awk -F '\t' 'NR > 1 && NF >= 4 { count++ } END { print count + 0 }' \
    "$OUT/sim_${arm}.tsv")
  echo "SIM_${arm}=$score rows=$rows" | tee -a "$VERDICT"
done

python3 - "$OUT/sim_control.tsv" "$OUT/sim_forced.tsv" <<'PY' | tee -a "$VERDICT"
import csv
import statistics
import sys


def load(path):
    rows = {}
    with open(path, newline="") as handle:
        for row in list(csv.reader(handle, delimiter="\t"))[1:]:
            if len(row) < 4:
                continue
            key = row[2].rsplit("/", 1)[-1]
            rows[key] = float(row[3])
    return rows


control = load(sys.argv[1])
forced = load(sys.argv[2])
if control.keys() != forced.keys():
    only_control = sorted(control.keys() - forced.keys())[:5]
    only_forced = sorted(forced.keys() - control.keys())[:5]
    raise SystemExit(
        f"SIM_ID_MISMATCH only_control={only_control} only_forced={only_forced}"
    )
deltas = [forced[key] - control[key] for key in control]
print(
    f"PAIRED_DELTA n={len(deltas)} mean={statistics.mean(deltas):.6f} "
    f"median={statistics.median(deltas):.6f} "
    f"improved={sum(value > 0 for value in deltas)} "
    f"equal={sum(value == 0 for value in deltas)} "
    f"worse={sum(value < 0 for value in deltas)}"
)
PY

echo "PAIRED_SIM_DONE $(date -u +%FT%TZ)" | tee -a "$VERDICT"
