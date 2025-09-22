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
    libxmlsec1-dev libffi-dev liblzma-dev 
#    && apt clean && rm -rf /var/lib/apt/lists/*
sudo apt-get install -y build-essential zlib1g-dev libffi-dev \
    libssl-dev libbz2-dev liblzma-dev libsqlite3-dev libreadline-dev \
    uuid-dev libgdbm-dev tk-dev

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


```bash
PYTHON_VERSION=3.13.7
curl -fsSLO https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
tar -xzf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}

# Native CPU, LTO, better linking
export CFLAGS="-O3 -pipe -march=native -mtune=native -fno-semantic-interposition -fno-plt"
export CPPFLAGS="-I/usr/include" 
export LDFLAGS="-Wl,-O3 -Wl,--as-needed -L/usr/lib"
# Ensure GCC LTO tools
export AR=gcc-ar RANLIB=gcc-ranlib NM=gcc-nm

# Optional: smaller binary, slight runtime gain for imports
# (removes many built-in docstrings; keep off if you need them)
EXTRA_CONF="--without-doc-strings"
# EXTRA_CONF=""

./configure \
  --prefix=/usr/local \
  --with-lto \
  --enable-optimizations \
  $EXTRA_CONF

# Faster, targeted PGO profile (instead of full test suite)
# from : https://bugs.python.org/issue36044
export PROFILE_TASK=-m test.regrtest --pgo \
        test_collections \
        test_dataclasses \
        test_difflib \
        test_embed \
        test_float \
        test_functools \
        test_generators \
        test_int \
        test_itertools \
        test_json \
        test_logging \
        test_long \
        test_ordered_dict \
        test_pickle \
        test_pprint \
        test_re \
        test_set \
        test_statistics \
        test_struct \
        test_tabnanny \
        test_xml_etree


make -j"$(nproc)"
sudo make altinstall

# Strip binaries and shared objects
sudo strip --strip-unneeded /usr/local/bin/python3.13 || true
sudo find /usr/local/lib/python3.13 -name "*.so" -type f -print0 | xargs -0 -r strip --strip-unneeded || true

# cd ..
# rm -f "Python-${PYTHON_VERSION}" "Python-${PYTHON_VERSION}.tgz"
```

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