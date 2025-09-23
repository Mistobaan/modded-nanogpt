export OMP_NUM_THREADS=24
export OMP_PROC_BIND=SPREAD
export OMP_PLACES=CORES

export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1      # often helps throughput

torchrun \
    --standalone \
    --nproc_per_node=8 \
    --no_python \
    bash -lc './train_gpt.sh'