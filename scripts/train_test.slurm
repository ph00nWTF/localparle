#!/bin/bash
#SBATCH --job-name=fr-test
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=0:30:00
#SBATCH --output=logs/train_test_%j.out
#SBATCH --error=logs/train_test_%j.out

# 15-minute single-A100 smoke test: 1% of train, 50 steps, no checkpointing.
# Validates preprocessing, model loading, and training steps on A100 (actual hardware).
# Success = log shows 50 steps with decreasing loss and no crash.

set -euo pipefail

PROJECT=/kuhpc/work/deng/e394n539/FrenchAI/french
cd "$PROJECT"

module load cuda/12.8

export HF_HOME=/kuhpc/scratch/deng/e394n539/hf_cache
export HF_DATASETS_CACHE=/kuhpc/scratch/deng/e394n539/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$PATH"

echo "Test job $SLURM_JOB_ID starting on $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
uv run python -c "import torch; print('torch', torch.__version__, '/ CUDA', torch.version.cuda, '/ GPUs:', torch.cuda.device_count())"

uv run python scripts/train.py --config configs/train_test.yaml

echo "Test job $SLURM_JOB_ID finished — check loss decreased over 50 steps"
