sudo apt install linux-headers-$(uname -r)
sudo apt -V install libnvidia-compute-580 nvidia-dkms-580
# sudo apt install ubuntu-drivers-common
# sudo ubuntu-drivers install --gpgpu nvidia:535-server # or the latest recommended version
# sudo apt install nvidia-utils-580-server

DEBIAN_FRONTEND=noninteractive
PATH=/usr/local/bin:$PATH

sudo apt update && apt install -y --no-install-recommends build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev 
#    && apt clean && rm -rf /var/lib/apt/lists/*

PYTHON_VERSION=3.12.7 curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
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