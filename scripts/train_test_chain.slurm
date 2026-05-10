#!/bin/bash
#SBATCH --job-name=fr-test-chain
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:l40:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:05:00
#SBATCH --output=logs/train_test_chain_%j.out
#SBATCH --error=logs/train_test_chain_%j.out

# Chain smoke test: 100 steps total, save every 30, only 5 minutes wall time.
# First job will hit timeout around step ~50-60, leaving checkpoint-30 and checkpoint-60.
# Auto-resubmit fires; second job loads latest checkpoint and finishes to step 100.
# Success criteria:
#   - Job 1 saves at least 1 checkpoint before timeout
#   - Job 2 logs "Loading model from checkpoint-XX" and continues from saved step
#   - Final model saved at /kuhpc/scratch/.../models/test-chain-final

set -euo pipefail

PROJECT=/kuhpc/work/deng/e394n539/FrenchAI/french
FINAL_MODEL=/kuhpc/scratch/deng/e394n539/french-data/models/test-chain-final
cd "$PROJECT"

module load cuda/12.8

export HF_HOME=/kuhpc/scratch/deng/e394n539/hf_cache
export HF_DATASETS_CACHE=/kuhpc/scratch/deng/e394n539/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$PATH"

echo "Chain test job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

uv run torchrun --nproc_per_node=2 scripts/train_v2.py --config configs/train_test_chain.yaml

echo "Job $SLURM_JOB_ID finished"

if [ ! -f "$FINAL_MODEL/model.safetensors" ] && [ ! -f "$FINAL_MODEL/pytorch_model.bin" ]; then
    echo "Chain test incomplete — resubmitting"
    sbatch "$PROJECT/scripts/train_test_chain.slurm"
fi
