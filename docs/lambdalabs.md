

## System
```
$ uname -ra
Linux 209-20-157-122 6.11.0-29-generic #29~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Jun 26 14:16:59 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
```

```bash
export PYTHON_VERSION=3.13
export PYTORCH_VERSION=2.8.0
export CUDA_VERSION=12.8
source /opt/miniconda/bin/activate
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n pytorch -c conda-forge python~=${PYTHON_VERSION} uv -y
conda activate pytorch
```

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install uv -y 
```

```bash
# download chezmoi
cd ~/
export GITHUB_USERNAME=Mistobaan
sh -c "$(curl -fsLS get.chezmoi.io)" -- init --apply $GITHUB_USERNAME
git clone https://github.com/Mistobaan/modded-nanogpt
cd modded-nanogpt
uv sync
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=modded-nanogpt
```