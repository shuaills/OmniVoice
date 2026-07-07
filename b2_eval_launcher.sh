#!/usr/bin/env bash
# B2 Seed-TTS eval generation: 8 shards, one GPU each.
# Env overrides: TSV (testset tsv), CKPT, OUT (wav out dir), LANG_HINT
set -euo pipefail
REPO_ROOT="/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice"
DONOR_VENV="/opt/gpfs/users/yinfeng/work/OmniVoice/.venv"
cd "${REPO_ROOT}"
source "${DONOR_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}"
python -c "import omnivoice, sys; p=omnivoice.__file__; print('omnivoice from:', p); sys.exit(0 if p.startswith('${REPO_ROOT}') else 1)"
TSV="${TSV:-/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset/zh/test.tsv}"
CKPT="${CKPT:-exp/block_b2/checkpoint-50000}"
BASE="/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block"
OUT="${OUT:-results_b2_50k/seedtts_zh}"
mkdir -p logs "${OUT}"
pids=()
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python tests/seedtts_blockwise_gen.py \
    --tsv "${TSV}" --ckpt "${CKPT}" --base "${BASE}" --out "${OUT}" \
    --shard ${i}/8 >> "logs/eval_$(basename ${OUT})_shard${i}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "ALL_SHARDS_DONE rc=${rc}"
exit ${rc}
