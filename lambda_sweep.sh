#!/usr/bin/env bash
# Overnight lambda_eos dose-response: x5 -> x20 -> x100, each 10k ft steps from R2-final,
# then zh300+en300 verdicts with runaway counts. Also evals exp3 (x1 control) final after arm 1.
set -u
L=/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice
MAIN=/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice
E=/opt/gpfs/users/shuai/work/block-emilia-parity/OmniVoice
DL=/opt/gpfs/users/yinfeng/work/OmniVoice
BASE=/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block
RES=$E/results_autoeval
LEDGER=$RES/ledger_splitloss.txt
TREND=$MAIN/results_wer_trend
cd $L
source $DL/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p logs

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

for mult in 5 20 100; do
  export PYTHONPATH="$L:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"
  accelerate launch --gpu_ids "$(seq -s, 0 7)" --num_processes 8 \
    -m omnivoice.cli.train \
    --train_config examples/config/train_config_ft10k_lx$mult.json \
    --data_config examples/config/data_config_emilia_full_blockparity.json \
    --output_dir exp/splitloss_ft10k_lambdaeos_x$mult 2>&1 | tee logs/sweep_lx$mult.log | tail -2
  echo "SWEEP_TRAIN_DONE x$mult rc=${PIPESTATUS[0]}"
  verdict_ck lx$mult $L/exp/splitloss_ft10k_lambdaeos_x$mult/checkpoint-10000
  if [ $mult -eq 5 ] && [ -d $L/exp/blockcausal_splitloss_ft15k_from_r2final/checkpoint-15000 ]; then
    verdict_ck lx1_control $L/exp/blockcausal_splitloss_ft15k_from_r2final/checkpoint-15000
  fi
done
echo "LAMBDA_SWEEP_ALL_DONE"
