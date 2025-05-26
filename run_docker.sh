#!/bin/bash

set -e

docker run \
    -it \
    --rm \
    --gpus all \
    -v $(pwd):/modded-nanogpt \
    -v $HF_HOME:/home/ubuntu/.cache/huggingface \
    -e HF_HOME=/home/ubuntu/.cache/huggingface \
    -e HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER \
    modded-nanogpt python data/cached_fineweb10B.py 8