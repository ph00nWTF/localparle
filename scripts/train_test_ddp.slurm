#!/bin/bash
#SBATCH --job-name=fr-test-ddp
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:a40:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/train_test_ddp_%j.out
#SBATCH --error=logs/train_test_ddp_%j.out

# DDP smoke test: 2× A40, 1% data, 50 steps, validates frozen_feature_extractor + DDP compatibility.
# This actually tests the distributed training pipeline (unlike single-GPU test).

set -euo pipefail

PROJECT=/kuhpc/work/deng/e394n539/FrenchAI/french
cd "$PROJECT"

module load cuda/12.8

export HF_HOME=/kuhpc/scratch/deng/e394n539/hf_cache
export HF_DATASETS_CACHE=/kuhpc/scratch/deng/e394n539/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$PATH"

echo "DDP test job $SLURM_JOB_ID starting on $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
uv run python -c "import torch; print('torch', torch.__version__, '/ CUDA', torch.version.cuda, '/ GPUs:', torch.cuda.device_count())"

uv run torchrun --nproc_per_node=2 scripts/train.py --config configs/train_test_ddp.yaml

echo "DDP test job $SLURM_JOB_ID finished — check loss decreased over 50 steps and no DDP errors"
