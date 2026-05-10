#!/bin/bash
#SBATCH --job-name=fr-wav2vec2-l40
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:l40:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --output=logs/train_l40_%j.out
#SBATCH --error=logs/train_l40_%j.out

set -uo pipefail

PROJECT=/kuhpc/work/deng/e394n539/FrenchAI/french
FINAL_MODEL=/kuhpc/scratch/deng/e394n539/french-data/models/final-v2
cd "$PROJECT"

# Always run resubmit check, even if torchrun crashes — set -e was killing the script
# before the if-block could resubmit on python crashes.
trap 'echo "Trap: training exited with code $?"; if [ ! -f "$FINAL_MODEL/model.safetensors" ] && [ ! -f "$FINAL_MODEL/pytorch_model.bin" ]; then echo "Final model not present — auto-resubmitting"; sbatch "$PROJECT/scripts/train_hpc_v2_l40.slurm"; fi' EXIT

module load cuda/12.8

export HF_HOME=/kuhpc/scratch/deng/e394n539/hf_cache
export HF_DATASETS_CACHE=/kuhpc/scratch/deng/e394n539/hf_cache/datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$PATH"

echo "L40 full job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

uv run torchrun --nproc_per_node=4 scripts/train_v2.py --config configs/train_v2.yaml

echo "Job $SLURM_JOB_ID finished"
# Auto-resubmit handled by EXIT trap above (fires on success AND crash)
