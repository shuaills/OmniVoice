#!/usr/bin/env bash
# B2 zh-subset sweep wave 2: locate CFG optimum (wave 1: 1.0=7.50, 2.0=3.45, 3.0=2.89).
set -uo pipefail
ROOT=/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice
TSVPATH=/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset/zh/test.tsv
SWEEP=${ROOT}/results_b2_sweep
run_cfg() {
  local tag=$1 steps=$2 gs=$3 dt=$4
  echo "=== CONFIG ${tag}: steps=${steps} gs=${gs} dtype=${dt} start $(date) ==="
  TSV="${TSVPATH}" OUT="${SWEEP}/${tag}/seedtts_zh" STEPS="${steps}" GS="${gs}" DTYPE="${dt}" LIMIT=38 \
    bash "${ROOT}/b2_eval_launcher.sh" || { echo "CONFIG ${tag} GEN FAILED"; return 1; }
  WAV_DIR="${SWEEP}/${tag}" TESTSET=seedtts_zh bash /opt/gpfs/users/yinfeng/work/tts_eval/eval.sh \
    || echo "CONFIG ${tag} EVAL FAILED"
  echo "=== CONFIG ${tag} done $(date) ==="
}
run_cfg gs2p5 16 2.5 fp16
run_cfg gs3p5 16 3.5 fp16
run_cfg gs4p0 16 4.0 fp16
echo "SWEEP2_ALL_DONE"
