#!/bin/bash

set -e 

sudo apt update && sudo apt install -y --no-install-recommends build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev libgdbm-dev libgdbm-compat-dev

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
export PROFILE_TASK="-m test.regrtest --pgo \
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
        test_xml_etree"


make -j"$(nproc)"
sudo make altinstall

# Strip binaries and shared objects
sudo strip --strip-unneeded /usr/local/bin/python3.13 || true
sudo find /usr/local/lib/python3.13 -name "*.so" -type f -print0 | sudo xargs -0 -r strip --strip-unneeded || true

# cd ..
# rm -f "Python-${PYTHON_VERSION}" "Python-${PYTHON_VERSION}.tgz"
```