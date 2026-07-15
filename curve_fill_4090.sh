#!/usr/bin/env bash
# Sparse curve backfill for 300k@lx20 on 3x4090: ckpts {60k,120k,180k,240k,300k},
# zh300+en300 subset verdicts -> ledger. Self-terminates after 300k ckpt (releases cards).
set -u
L=/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice
MAIN=/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice
E=/opt/gpfs/users/shuai/work/block-emilia-parity/OmniVoice
DL=/opt/gpfs/users/yinfeng/work/OmniVoice
BASE=/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block
RES=$E/results_autoeval
LEDGER=$RES/ledger_splitloss.txt
TREND=$MAIN/results_wer_trend
EXP=$L/exp/blockcausal_splitloss_emilia_300k_lx20
MARK=$EXP/.evaled_4090; mkdir -p $MARK
NG=3
source $DL/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

verdict_ck() { # tag ckdir
  local tag=$1 ckdir=$2
  cd $MAIN; export PYTHONPATH=$MAIN
  local SHIM=$RES/shim_$tag; mkdir -p $SHIM
  cp $MAIN/results_scratch_eosdecouple_50k/shim_ckpt/config.json $SHIM/
  for f in model.safetensors tokenizer.json tokenizer_config.json train_config.json; do
    ln -sfn $ckdir/$f $SHIM/$f
  done
  for lang in zh en; do
    local TSV=$TREND/first300.tsv JSONL=$TREND/first300.jsonl
    [ $lang = en ] && TSV=$RES/en_first300.tsv && JSONL=$RES/en_first300.jsonl
    local OUT=$RES/${tag}_${lang}_wavs; mkdir -p $OUT; local pids=()
    for i in $(seq 0 $((NG-1))); do
      CUDA_VISIBLE_DEVICES=$i python tests/seedtts_blockwise_gen.py \
        --tsv $TSV --ckpt $SHIM --base $BASE --out $OUT \
        --steps-per-block 16 --guidance-scale 2.0 --dtype bf16 \
        --shard $i/$NG > $RES/gen_${tag}_${lang}_sh$i.log 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p; done
    python omnivoice/eval/wer/seedtts.py --wav-path $OUT --test-list $JSONL \
      --model-dir $DL/download/tts_eval_models --lang $lang \
      --decode-path $RES/wer_${tag}_$lang.tsv --batch-size 4 > $RES/wer_${tag}_$lang.sublog 2>&1
    local W=$(grep -oE "Seed-TTS WER \(Avg of WERs\): [0-9.]+%" $RES/wer_${tag}_$lang.sublog | tail -1)
    local RA=$(python3 -c "
import csv
rows = [r for r in csv.reader(open(\"$RES/wer_${tag}_$lang.tsv\"), delimiter=\"\t\") if len(r) >= 2][1:]
print(sum(1 for r in rows if float(r[1]) > 0.5))
" 2>/dev/null)
    echo "SWEEP $tag $lang ${W:-FAILED} runaway=${RA:-?} $(date +%F_%T)" >> $LEDGER
  done
}

for st in 100000 110000 130000 140000 150000 160000 170000 180000 190000 200000 210000 220000 230000 250000 260000 270000 280000 290000 300000; do
  [ -f $MARK/$st ] && continue
  for i in $(seq 1 720); do [ -d $EXP/checkpoint-$st ] && break; sleep 60; done
  if [ ! -d $EXP/checkpoint-$st ]; then echo "GIVEUP_WAIT_$st"; continue; fi
  sleep 60
  verdict_ck sl300klx20_$st $EXP/checkpoint-$st && touch $MARK/$st
  echo "CURVE_DONE_$st $(date +%F_%T)"
done
echo "CURVE_BACKFILL_ALL_DONE $(date +%F_%T)"
