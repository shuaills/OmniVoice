#!/usr/bin/env bash
# Final main-table run at swept-optimal CFG: full zh (2020) + full en (1088), gen + score.
# Usage: bash b2_main_final.sh <guidance_scale>
set -uo pipefail
GSOPT="${1:?usage: b2_main_final.sh <guidance_scale>}"
ROOT=/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice
DS=/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset
TAG="gs${GSOPT/./p}"
OUTROOT="${ROOT}/results_b2_final_${TAG}"
for lang in zh en; do
  echo "=== FINAL ${lang} gs=${GSOPT} start $(date) ==="
  TSV="${DS}/${lang}/test.tsv" OUT="${OUTROOT}/seedtts_${lang}" GS="${GSOPT}" \
    bash "${ROOT}/b2_eval_launcher.sh" || { echo "FINAL ${lang} GEN FAILED"; exit 1; }
done
WAV_DIR="${OUTROOT}" TESTSET=seedtts bash /opt/gpfs/users/yinfeng/work/tts_eval/eval.sh \
  || echo "FINAL EVAL FAILED"
echo "FINAL_MAIN_TABLE_DONE gs=${GSOPT}"
