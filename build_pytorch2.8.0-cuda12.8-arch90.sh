
export PYTHON_PATH=/usr/local/bin/python3.13
export CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)')
export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST="9.0"
export BUILD_TEST=0
export USE_MKLDNN=1
export USE_XNNPACK=1
export USE_NNPACK=0
export PYTORCH_BUILD_VERSION=2.5.0   # adjust if needed
export PYTORCH_BUILD_NUMBER=1

# USE_SYSTEM_NVTX

# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

$PYTHON_PATH -m venv .venv

source .venv/bin/activate

# Run this command from the PyTorch directory after cloning the source code using the “Get the PyTorch Source“ section above
sudo apt install cmake ninja-build -y

make triton

export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install --no-build-isolation -v -e .
pip install mkl-static mkl-include nvtx

sudo apt install -y libcufile-dev
python setup.py bdist_wheel
pip install dist/torch-*.whl