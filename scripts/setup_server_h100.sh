# Install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate

# Create conda environment
conda create -n llm python=3.10 -y
conda activate llm

# Install pytorch
pip install -r requirements.txt

# Visit https://developer.nvidia.com/cuda-downloads and download CUDA Toolkit
# Check compatibility matrix: https://docs.nvidia.com/deploy/cuda-compatibility/
# Examples:
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt-get update
# sudo apt-get -y install cuda-toolkit-12-6

# Install flash-attn
pip install flash-attn --no-build-isolation
