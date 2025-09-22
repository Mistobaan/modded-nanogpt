sudo apt install linux-headers-$(uname -r)
sudo apt -V install libnvidia-compute-580 nvidia-dkms-580
# sudo apt install ubuntu-drivers-common
# sudo ubuntu-drivers install --gpgpu nvidia:535-server # or the latest recommended version
# sudo apt install nvidia-utils-580-server
# https://www.nvidia.com/en-us/drivers/details/246852/
# Data Center Driver for Ubuntu 24.04 575.57.08 | Linux 64-bit Ubuntu 24.04

# https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/
NVIDA_DRIVER_VERSION=575
sudo apt install nvidia-utils-575
sudo apt install nvidia-fabricmanager-${NVIDA_DRIVER_VERSION} libnvidia-nscq-${NVIDA_DRIVER_VERSION}

DEBIAN_FRONTEND=noninteractive
PATH=/usr/local/bin:$PATH

sudo apt update && apt install -y --no-install-recommends build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev libgdbm-dev libgdbm-compat-dev
# sudo apt-get install -y build-essential zlib1g-dev libffi-dev \
#     libssl-dev libbz2-dev liblzma-dev libsqlite3-dev libreadline-dev \
#     uuid-dev libgdbm-dev tk-dev

pip install torch --index-url https://download.pytorch.org/whl/nightly/cu129 --force


PYTHON_VERSION=3.13.5 curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129


sudo ln -s /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.12 /usr/local/bin/pip



```
export TORCH_CUDA_ARCH_LIST=9.0
export PATH=/usr/local/cuda-12.9/bin:$PATH
```



useful commands
```bash
sudo apt remove --purge '^nvidia-.*'
```


```bash
sudo ubuntu-drivers --gpgpu install
```