#!/bin/bash
#SBATCH --job-name=fr-wav2vec2-v2
#SBATCH --partition=sixhour
#SBATCH --nodelist=r32r05n01
#SBATCH --gres=gpu:a40:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --output=logs/train_v2_%j.out
#SBATCH --error=logs/train_v2_%j.out

set -euo pipefail

PROJECT=/kuhpc/work/deng/e394n539/FrenchAI/french
FINAL_MODEL=/kuhpc/scratch/deng/e394n539/french-data/models/final-v2
cd "$PROJECT"

module load cuda/12.8

export HF_HOME=/kuhpc/scratch/deng/e394n539/hf_cache
export HF_DATASETS_CACHE=/kuhpc/scratch/deng/e394n539/hf_cache/datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$PATH"

echo "Job $SLURM_JOB_ID starting on $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
uv run python -c "import torch; print('torch', torch.__version__, '/ CUDA', torch.version.cuda, '/ GPUs:', torch.cuda.device_count())"

uv run torchrun --nproc_per_node=4 scripts/train_v2.py --config configs/train_v2.yaml

echo "Job $SLURM_JOB_ID finished"

if [ ! -f "$FINAL_MODEL/model.safetensors" ] && [ ! -f "$FINAL_MODEL/pytorch_model.bin" ]; then
    echo "Training incomplete — resubmitting"
    sbatch "$PROJECT/scripts/train_hpc_v2.slurm"
fi
