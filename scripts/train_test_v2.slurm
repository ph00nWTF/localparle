#!/bin/bash
#SBATCH --job-name=fr-test-v2
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:a40:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/train_test_v2_%j.out
#SBATCH --error=logs/train_test_v2_%j.out

set -euo pipefail

PROJECT=/kuhpc/work/deng/e394n539/FrenchAI/french
cd "$PROJECT"

module load cuda/12.8

export HF_HOME=/kuhpc/scratch/deng/e394n539/hf_cache
export HF_DATASETS_CACHE=/kuhpc/scratch/deng/e394n539/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$PATH"

echo "DDP test v2 job $SLURM_JOB_ID on $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
uv run python -c "import torch; print('torch', torch.__version__, '/ CUDA', torch.version.cuda, '/ GPUs:', torch.cuda.device_count())"

uv run torchrun --nproc_per_node=2 scripts/train_v2.py --config configs/train_test_v2.yaml

echo "DDP test v2 job $SLURM_JOB_ID finished"
