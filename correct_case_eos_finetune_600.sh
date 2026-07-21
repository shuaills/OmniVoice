#!/usr/bin/env bash
# Continue the plain correct-case run from step 300 to total step 600.
set -Eeuo pipefail

ROOT=${SOURCE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}
RUNTIME_VENV=${RUNTIME_VENV:-/opt/gpfs/users/yinfeng/work/OmniVoice/.venv}
OUTPUT_ROOT=${OUTPUT_ROOT:-/opt/gpfs/users/shuai/experiments/eos-correct-case-20260721}
RUN_ID=${RUN_ID:-ce600-v1}
RESUME_CHECKPOINT=/opt/gpfs/users/shuai/experiments/eos-correct-case-20260721/ce300-v1/train/checkpoint-300
EXPECTED_RESUME_MODEL_SHA256=7c22969c3b92675bb1bf5155c47a0eac6616ccd8c28273fa04b00afae910e2a9
TRAIN_CONFIG=$ROOT/examples/config/train_config_correct_case_eos_600.json
DATA_CONFIG=$ROOT/examples/config/data_config_emilia_full_blockparity.json
RUN_ROOT=$OUTPUT_ROOT/$RUN_ID

[[ -x $RUNTIME_VENV/bin/python ]] || { echo "missing runtime: $RUNTIME_VENV" >&2; exit 2; }
[[ -d $RESUME_CHECKPOINT ]] || { echo "missing checkpoint: $RESUME_CHECKPOINT" >&2; exit 2; }
[[ -f ${RESUME_CHECKPOINT%/train/checkpoint-300}/PASS ]] || {
  echo "missing source PASS marker" >&2
  exit 2
}
for artifact in model.safetensors optimizer.bin scheduler.bin random_states_0.pkl random_states_1.pkl; do
  [[ -s $RESUME_CHECKPOINT/$artifact ]] || {
    echo "incomplete resume checkpoint: $RESUME_CHECKPOINT/$artifact" >&2
    exit 2
  }
done
actual_resume_model_sha256=$(sha256sum "$RESUME_CHECKPOINT/model.safetensors" | awk '{print $1}')
[[ $actual_resume_model_sha256 == "$EXPECTED_RESUME_MODEL_SHA256" ]] || {
  echo "wrong resume model hash: expected=$EXPECTED_RESUME_MODEL_SHA256 actual=$actual_resume_model_sha256" >&2
  exit 2
}
[[ -f $TRAIN_CONFIG && -f $DATA_CONFIG ]] || { echo "missing config" >&2; exit 2; }
[[ ! -e $RUN_ROOT ]] || { echo "refusing to reuse output: $RUN_ROOT" >&2; exit 2; }

cd "$ROOT"
actual_commit=$(git rev-parse HEAD)
[[ $actual_commit == "$EXPECTED_COMMIT" ]] || {
  echo "wrong source commit: expected=$EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
}
[[ -z $(git status --porcelain --untracked-files=all) ]] || {
  echo "source checkout is not clean" >&2
  exit 2
}

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed '/^$/d' | wc -l | tr -d ' ')
[[ $gpu_count == 2 ]] || { echo "expected 2 GPUs, got $gpu_count" >&2; exit 2; }

mkdir -p "$RUN_ROOT/logs"
export PYTHONNOUSERSITE=1
export PYTHONPATH=$ROOT:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$RUNTIME_VENV/bin/python" scripts/check_checkpoint_vocab.py \
  --checkpoint "$RESUME_CHECKPOINT"
"$RUNTIME_VENV/bin/python" scripts/check_correct_case_eos_config.py \
  "$TRAIN_CONFIG"

"$RUNTIME_VENV/bin/python" -m accelerate.commands.accelerate_cli launch \
  --num_processes 2 --num_machines 1 --mixed_precision bf16 --gpu_ids 0,1 \
  -m omnivoice.cli.train \
  --train_config "$TRAIN_CONFIG" \
  --data_config "$DATA_CONFIG" \
  --output_dir "$RUN_ROOT/train" \
  2>&1 | tee "$RUN_ROOT/logs/train.log"

grep -Fq "Resumed from step 300" "$RUN_ROOT/logs/train.log" || {
  echo "training log does not prove resume from step 300" >&2
  exit 2
}
checkpoint=$RUN_ROOT/train/checkpoint-600
for artifact in model.safetensors config.json train_config.json optimizer.bin scheduler.bin random_states_0.pkl random_states_1.pkl; do
  [[ -s $checkpoint/$artifact ]] || { echo "missing artifact: $checkpoint/$artifact" >&2; exit 2; }
done
printf 'CORRECT_CASE_EOS_FINETUNE_600_PASS\n' > "$RUN_ROOT/PASS"
