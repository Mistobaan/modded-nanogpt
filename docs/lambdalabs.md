

## System
```
$ uname -ra
Linux 209-20-157-122 6.11.0-29-generic #29~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Jun 26 14:16:59 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
```

export PYTHON_VERSION=3.13
export PYTORCH_VERSION=2.8.0
export CUDA_VERSION=12.8
source /opt/miniconda/bin/activate
conda create -n pytorch python~=${PYTHON_VERSION} -y
conda activate pytorch
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install uv -y 

uv sync

uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=nanogpt-modded