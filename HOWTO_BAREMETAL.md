# How to install top performance baremetal

Ensure you are on the right architecture
```bash
$ uname
Linux 192-222-52-75 6.8.0-83-generic #83-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  5 21:46:54 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
```

Ensure you have the gpus

```bash
lspci | grep -i nvidia
```

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9
```

## Reference:
- [CUDA toolkit 12.9](https://developer.nvidia.com/cuda-12-9-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)


/usr/local/bin/python3.13 -m venv  .venv 
source .venv/bin/activate

 ./setup_8xH100.sh /home/ubuntu/fs-us-south-2