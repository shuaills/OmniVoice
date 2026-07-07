#!/usr/bin/env bash
# Full en set at steps-per-block=32 (main-table-grade steps knife), then score.
set -uo pipefail
ROOT=/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice
TSV=/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset/en/test.tsv \
  OUT="${ROOT}/results_b2_steps32/seedtts_en" STEPS=32 \
  bash "${ROOT}/b2_eval_launcher.sh" || { echo "GEN FAILED"; exit 1; }
WAV_DIR="${ROOT}/results_b2_steps32" TESTSET=seedtts_en \
  bash /opt/gpfs/users/yinfeng/work/tts_eval/eval.sh || echo "EVAL FAILED"
echo "EN_STEPS32_DONE"
