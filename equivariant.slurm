#!/bin/bash
#SBATCH --job-name=equivCNN-a100
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=equivariantCNN.log
#SBATCH --error=equivariantCNN.log

export PYTHONUNBUFFERED=1

echo "🌱 Setting up environment..."

# Load miniconda
module load miniconda
source /vast/palmer/apps/avx2/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate 452final_env

# Load a GCC module (important!)
module load GCCcore/12.2.0  # <- Picked one version available on Grace
pip install escnn

# Now run your code
python ~/cpsc452proj/run.py
