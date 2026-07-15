#!/usr/bin/env bash
# 16-GPU (2 nodes x 8) from-scratch 300k @ lambda_eos x20. Same GLOBAL batch as 8-GPU recipe:
# per-GPU batch_tokens 15648 -> 7824, so 16 x 7824 == 8 x 15648 (step-for-step comparable).
# Node rendezvous via GPFS file (no DNS assumption). Guardian shell on both nodes.
set -u
L=/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice
DL=/opt/gpfs/users/yinfeng/work/OmniVoice
RDV=$L/logs/16g_master_ip.txt
cd $L
source $DL/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$L:/opt/gpfs/users/shuai/work/block-b2-perf/pylibs"
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
mkdir -p logs

HN=$(hostname)
NODE_RANK=${HN##*-}
echo "host=$HN node_rank=$NODE_RANK"
if [ "$NODE_RANK" = "0" ]; then
  hostname -i | awk '{print $1}' > $RDV.tmp && mv $RDV.tmp $RDV
else
  for i in $(seq 1 120); do [ -s $RDV ] && break; sleep 5; done
fi
MASTER_IP=$(cat $RDV)
echo "master_ip=$MASTER_IP"

accelerate launch --num_machines 2 --machine_rank $NODE_RANK \
  --main_process_ip $MASTER_IP --main_process_port 29517 \
  --num_processes 16 --gpu_ids "$(seq -s, 0 7)" \
  -m omnivoice.cli.train \
  --train_config examples/config/train_config_emilia_splitloss_300k_lx20_16g.json \
  --data_config examples/config/data_config_emilia_full_blockparity.json \
  --output_dir exp/blockcausal_splitloss_emilia_300k_lx20_16g 2>&1 | tee logs/splitloss_300k_lx20_16g_node$NODE_RANK.log | tail -2
echo "SPLITLOSS_300K_LX20_16G_EXITED node=$NODE_RANK rc=${PIPESTATUS[0]} $(date +%F_%T)"
sleep infinity
