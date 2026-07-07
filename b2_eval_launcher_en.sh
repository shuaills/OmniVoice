#!/usr/bin/env bash
export TSV=/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset/en/test.tsv
export OUT=results_b2_50k/seedtts_en
exec bash /opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice/b2_eval_launcher.sh
