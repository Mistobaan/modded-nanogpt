#!/usr/bin/env bash
set -euo pipefail

RANK=${RANK:-0}
LOCAL_RANK=${LOCAL_RANK:-0}

# GPU mapping: rank == GPU id
# export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
#echo $CUDA_VISIBLE_DEVICES

# NUMA/NODE binding. adjust these values based on `nvidia-smi topo -m`
if [ "${LOCAL_RANK}" -le 3 ]; then
  NODE=0;  MEM=0
  
else
  NODE=1; MEM=1
fi

echo "starting process ${RANK} on node ${NODE}/${MEM}"

exec numactl --cpunodebind="${NODE}" --membind="${MEM}" \
  python train_gpt.py