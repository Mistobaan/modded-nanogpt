export OMP_NUM_THREADS=24
export OMP_PROC_BIND=SPREAD
export OMP_PLACES=CORES

torchrun \
    --standalone \
    --nproc_per_node=8 \
    --no_python \
    bash -lc './train_gpt.sh'