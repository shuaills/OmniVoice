#!/usr/bin/env bash
# B2 zh-subset knife sweep: steps-per-block / CFG scale / dtype.
# Subset = first 304 tsv rows (LIMIT=38 x 8 shards). Anchor (steps16 gs2.0 fp16)
# comes from results_b2_50k_v2 (identical seeds/config) - not regenerated.
set -uo pipefail
ROOT=/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice
TSVPATH=/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset/zh/test.tsv
SWEEP=${ROOT}/results_b2_sweep
mkdir -p "${SWEEP}"
run_cfg() {
  local tag=$1 steps=$2 gs=$3 dt=$4
  echo "=== CONFIG ${tag}: steps=${steps} gs=${gs} dtype=${dt} start $(date) ==="
  TSV="${TSVPATH}" OUT="${SWEEP}/${tag}/seedtts_zh" STEPS="${steps}" GS="${gs}" DTYPE="${dt}" LIMIT=38 \
    bash "${ROOT}/b2_eval_launcher.sh" || { echo "CONFIG ${tag} GEN FAILED"; return 1; }
  WAV_DIR="${SWEEP}/${tag}" TESTSET=seedtts_zh bash /opt/gpfs/users/yinfeng/work/tts_eval/eval.sh \
    || echo "CONFIG ${tag} EVAL FAILED"
  echo "=== CONFIG ${tag} done $(date) ==="
}
run_cfg steps32 32 2.0 fp16
run_cfg gs1p0   16 1.0 fp16
run_cfg gs3p0   16 3.0 fp16
run_cfg gs0     16 0   fp16
run_cfg fp32    16 2.0 fp32
echo "SWEEP_ALL_DONE"
