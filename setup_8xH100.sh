#!/bin/bash

# Check if the argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_storage>"
  exit 1
fi


PATH_STORAGE=$1

# Ensure PATH_STORAGE is an absolute path
if [[ "$PATH_STORAGE" != /* ]]; then
    echo "Error: PATH_STORAGE must be an absolute path."
    exit 1
fi

# Check if PATH_STORAGE exists and is a directory
if [ ! -d "$PATH_STORAGE" ]; then
    echo "Error: PATH_STORAGE directory does not exist: $PATH_STORAGE"
    exit 1
fi

# Exit immediately if a command exits with a non-zero status
set -e
# normalize path to avoid double slashes

# Create huggingface cache directory and set environment variables
mkdir -p $(realpath ${PATH_STORAGE})/.cache/huggingface

echo "export HF_HOME=\"${PATH_STORAGE}/.cache/huggingface\"" >> ~/.bashrc
echo 'export HF_HUB_ENABLE_HF_TRANSFER="1"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/lib/cuda/lib64"' >> ~/.bashrc
echo 'export CUDA_HOME="/usr/lib/nvidia-cuda-toolkit"' >> ~/.bashrc


# # Check if Miniconda is already installed
# if [ ! -d "/home/ubuntu/miniconda3" ]; then
#     # Download and install Miniconda
#     # Detect system architecture and download appropriate Miniconda installer
#     ARCH=$(uname -m)
#     if [ "$ARCH" = "x86_64" ]; then
#       MINICONDA_ARCH="Linux-x86_64"
#     elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
#       MINICONDA_ARCH="Linux-aarch64"
#     else
#       echo "Unsupported architecture: $ARCH"
#       exit 1
#     fi
#     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-$MINICONDA_ARCH.sh -O /tmp/miniconda.sh
#     bash /tmp/miniconda.sh -b -p /home/ubuntu/miniconda3 -u
#     /home/ubuntu/miniconda3/bin/conda init bash
#     echo 'export PATH="/home/ubuntu/miniconda3/bin:$PATH"' >> ~/.bashrc
#     source ~/.bashrc
# else
#     /home/ubuntu/miniconda3/bin/conda init bash
#     echo 'export PATH="/home/ubuntu/miniconda3/bin:$PATH"' >> ~/.bashrc
#     source ~/.bashrc
# fi

# # Configure Conda and create PyTorch environment
# # Check if the pytorch environment exists and remove it if found
# if /home/ubuntu/miniconda3/bin/conda env list | grep -q "pytorch"; then
#   echo "Removing existing pytorch environment..."
#   /home/ubuntu/miniconda3/bin/conda remove -n pytorch --all -y
# fi
# /home/ubuntu/miniconda3/bin/conda install python~=3.12.7 -y
# /home/ubuntu/miniconda3/bin/conda install -c conda-forge libstdcxx-ng -y
# /home/ubuntu/miniconda3/bin/conda create -n pytorch python~=3.12 -y
# echo 'conda activate pytorch' >> ~/.bashrc

# # Install PyTorch
# /home/ubuntu/miniconda3/bin/conda run -n pytorch conda install torch~=2.7.0 torchvision torchaudio torch-cuda=12.8 -c pytorch -c nvidia -y
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# # Verify PyTorch installation
# /home/ubuntu/miniconda3/bin/conda run -n pytorch python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# # Add Git LFS repository and install
# sudo add-apt-repository -y ppa:git-core/ppa
# sudo apt-get update
# sudo apt-get install -y git-lfs
# git lfs install --skip-repo
