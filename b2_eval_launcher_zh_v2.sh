#!/usr/bin/env bash
export TSV=/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset/zh/test.tsv
export OUT=results_b2_50k_v2/seedtts_zh
exec bash /opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice/b2_eval_launcher.sh
