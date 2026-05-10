#!/bin/bash
#SBATCH --job-name=fr-eval
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.out

set -uo pipefail

PROJECT=/kuhpc/work/deng/e394n539/FrenchAI/french
MODEL=/kuhpc/scratch/deng/e394n539/french-data/models/final-v2
cd "$PROJECT"

module load cuda/12.8

export HF_HOME=/kuhpc/scratch/deng/e394n539/hf_cache
export HF_DATASETS_CACHE=/kuhpc/scratch/deng/e394n539/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$PATH"

echo "Eval job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

uv run python scripts/evaluate.py --model-dir "$MODEL" --config configs/train_v2.yaml --batch-size 16

echo "Eval $SLURM_JOB_ID finished"
