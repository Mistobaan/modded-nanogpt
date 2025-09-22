#!/bin/bash

set -e

NPROC_PER_NODE=8

# Check if the argument is provided
if [[ -n "$1" && "$1" =~ ^[1-9][0-9]*$ ]]; then
    NPROC_PER_NODE="$1"
else
    if [ -n "$1" ]; then
        echo "Error: Argument must be a positive integer." >&2
        exit 1
    fi
fi

echo NPROC_PER_NODE=$NPROC_PER_NODE

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train_gpt.py
