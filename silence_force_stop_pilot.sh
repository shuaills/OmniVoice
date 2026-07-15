#!/usr/bin/env bash
# Generated-token calibration for the silence-run force-stop fallback.
# One self-contained 3x4090 job; no wait loop and no guardian shell.
set -euo pipefail

C=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/OmniVoice
MAIN=/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice
E=/opt/gpfs/users/shuai/work/block-emilia-parity/OmniVoice
L=/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice
DL=/opt/gpfs/users/yinfeng/work/OmniVoice
BASE=/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block
CK=$L/exp/blockcausal_splitloss_emilia_300k_lx20/checkpoint-300000
RES=/opt/gpfs/users/shuai/work/silence-force-stop-campaign/results/pilot_en300_20260716
TOK=$RES/tokens
OUT=$RES/baseline_wavs
SHIM=$RES/shim
TSV=$E/results_autoeval/en_first300.tsv

cd "$C"
source "$DL/.venv/bin/activate"
export PYTHONPATH="$C"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RES/logs" "$TOK" "$OUT" "$SHIM"

cp "$MAIN/results_scratch_eosdecouple_50k/shim_ckpt/config.json" "$SHIM/"
for file in model.safetensors tokenizer.json tokenizer_config.json train_config.json; do
  ln -sfn "$CK/$file" "$SHIM/$file"
done

pids=()
for gpu in 0 1 2; do
  DUMP_TOKENS_DIR="$TOK" CUDA_VISIBLE_DEVICES=$gpu \
    python tests/seedtts_blockwise_gen.py \
      --tsv "$TSV" --ckpt "$SHIM" --base "$BASE" --out "$OUT" \
      --steps-per-block 16 --guidance-scale 2.0 --dtype bf16 \
      --shard "$gpu/3" \
      > "$RES/logs/gen_shard${gpu}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
if [[ $rc -ne 0 ]]; then
  echo "PILOT_GENERATION_FAILED"
  exit 1
fi

python tools/silence_token_audit.py \
  --token-dir "$TOK" \
  --meta-glob "$OUT/gen_meta_shard*.jsonl" \
  --run-frames 25 \
  --output "$RES/summary.json" \
  > "$RES/audit.log"

echo "SILENCE_FORCE_STOP_PILOT_DONE $(date -u +%FT%TZ)"
cat "$RES/summary.json"
