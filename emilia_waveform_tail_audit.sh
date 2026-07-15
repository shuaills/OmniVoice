#!/usr/bin/env bash
# CPU-only spread-sample audit; self-terminates and holds no GPU.
set -euo pipefail

C=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/OmniVoice
DL=/opt/gpfs/users/yinfeng/work/OmniVoice
PROV=/opt/gpfs/users/shuai/work/block-longform/OmniVoice/data/emilia_full/tokq/done
OUT=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/results/emilia_waveform_tail_1000

cd "$C"
source "$DL/.venv/bin/activate"
export PYTHONPATH="$C"
mkdir -p "$OUT"
python data_audit/emilia_waveform_tail_audit.py \
  --provenance-root "$PROV" \
  --sources-per-chunk 1 \
  --samples-per-source 50 \
  --workers 4 \
  --margins-db 25,30,35 \
  --output-dir "$OUT" \
  > "$OUT/run.log"
echo "EMILIA_WAVEFORM_TAIL_AUDIT_DONE $(date -u +%FT%TZ)"
cat "$OUT/summary.json"
