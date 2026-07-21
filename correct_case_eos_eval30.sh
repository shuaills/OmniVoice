#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export C=$ROOT
export BANDCTL_CK=/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block
export BAND4_CK=/opt/gpfs/users/shuai/experiments/eos-correct-case-20260721/ce300-v1/train/checkpoint-300
export EXPECTED_COUNT=30
export GPU_IDS=0,1,2
export GUIDANCE_SCALE=1.0
export RESULT_ROOT=/opt/gpfs/users/shuai/experiments/eos-correct-case-20260721/eval-first30
export RUN_ID=${RUN_ID:-base-vs-ce300-g1-$(date -u +%Y%m%dT%H%M%SZ)}

exec bash "$ROOT/band_ft10k_eval_pair.sh"
