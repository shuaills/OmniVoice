#!/usr/bin/env bash
# B2G on 7 GPUs (gpu3 squeeze: one 1-GPU pod blocks full-8 placement; 7/8 global batch, footnote in RESULTS).
export NUM_GPUS=7
exec bash /opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice/b2g_launcher.sh
