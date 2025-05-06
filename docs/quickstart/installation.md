# Installation
## Requirements
- **Python:** >= 3.9
- **CUDA:** >= 12.1

## Install
### Install from script (quickest)
We provide a script to install `RAGEN` on **Linux**. This script will install all the dependencies and set up the **Conda** environment for you.
```bash
# Clone the repository
git clone https://github.com/ZihanWang314/RAGEN.git

# Change directory to RAGEN
cd RAGEN

# Setup and download the dataset
bash scripts/setup_ragen.sh
```

### Note
If `setup_ragen.sh` fails, you can follow the manual setup instructions in `scripts/setup_ragen.md`.

### Install from source
We also provide manual installation command lines. We recommend you to use **Conda** for environment management:
```bash
# Create and activate a new conda environment
conda create -n ragen python=3.9 -y
conda activate ragen
```

For the best development experience, we recommend installing from the source code. This allows you to easily modify the code and contribute to the project.

```bash
# Clone the repository (if not done already)
git clone https://github.com/ZihanWang314/RAGEN.git
cd RAGEN

# Install the package in editable mode
pip install -e .
```
Then install the dependencies:
```bash
# Install PyTorch (with CUDA support if available)
# For CUDA 12.1:
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# For CPU only:
# pip install torch==2.4.0

# Optional: Install flash-attention (CUDA only)
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
pip3 install flash-attn --no-build-isolation

# Install remaining requirements
pip install -r requirements.txt
```

## Dataset
If you would like to test on our agent environments ([Sokoban](../practical_guide/examples/sokoban.md), [Bi-arm bandit](../practical_guide/examples/bi_arm_bandit.md), [frozenlake](../practical_guide/examples/frozenlake.md)), you can either download the dataset from our repository or prepare using our script.
### Download the dataset
Notice that `setup_ragen.sh` will automatically download the dataset for you. You can also run the following command manually to download the dataset (~ 7MB):
```bash
# Download the dataset
# This will download the dataset to the `data` directory
python scripts/download_dataset.py
```