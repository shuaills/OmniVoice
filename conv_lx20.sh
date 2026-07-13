#!/usr/bin/env bash
# Conv-arm redo with lambda_eos x20 (0.03728): R1-final migrated -> 50k @3e-5 cosine.
# Then bilingual zh300/en300 verdicts at 10k/30k/50k (ckpt keeper defeats rotation).
set -u
L=/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice
MAIN=/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice
E=/opt/gpfs/users/shuai/work/block-emilia-parity/OmniVoice
DL=/opt/gpfs/users/yinfeng/work/OmniVoice
BASE=/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block
RES=$E/results_autoeval
LEDGER=$RES/ledger_splitloss.txt
TREND=$MAIN/results_wer_trend
EXP=$L/exp/blockcausal_splitloss_conv50k_from_r1final_lx20
KEEP=$EXP/keep_ckpts
cd $L
source $DL/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p logs $KEEP

# checkpoint keeper: hardlink-copy 10k/30k as they appear so rotation can't eat them
(
  for st in 10000 30000; do
    while [ ! -d $EXP/checkpoint-$st ]; do sleep 300; done
    sleep 60
    [ -d $KEEP/checkpoint-$st ] || cp -al $EXP/checkpoint-$st $KEEP/checkpoint-$st
    echo "KEEPER saved checkpoint-$st $(date +%F_%T)"
  done
) >> logs/conv_lx20_keeper.log 2>&1 &
KEEPER_PID=$!

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
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python tests/seedtts_blockwise_gen.py \
        --tsv $TSV --ckpt $SHIM --base $BASE --out $OUT \
        --steps-per-block 16 --guidance-scale 2.0 --dtype bf16 \
        --shard $i/8 > $RES/gen_${tag}_${lang}_sh$i.log 2>&1 &
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
  cd $L
}

export PYTHONPATH="$L:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"
accelerate launch --gpu_ids "$(seq -s, 0 7)" --num_processes 8 \
  -m omnivoice.cli.train \
  --train_config examples/config/train_config_emilia_splitloss_conv50k_lx20.json \
  --data_config examples/config/data_config_emilia_full_blockparity.json \
  --output_dir exp/blockcausal_splitloss_conv50k_from_r1final_lx20 2>&1 | tee logs/splitloss_conv50k_lx20.log | tail -2
echo "CONV_LX20_TRAIN_DONE rc=${PIPESTATUS[0]}"
kill $KEEPER_PID 2>/dev/null

for st in 10000 30000 50000; do
  CK=$EXP/checkpoint-$st
  [ -d $CK ] || CK=$KEEP/checkpoint-$st
  if [ -d $CK ]; then
    verdict_ck convlx20_$st $CK
  else
    echo "SWEEP convlx20_$st MISSING_CKPT $(date +%F_%T)" >> $LEDGER
  fi
done
echo "CONV_LX20_ALL_DONE $(date +%F_%T)"
