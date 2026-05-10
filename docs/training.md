# Training Reproduction Guide

End-to-end recipe to take this repo from a fresh checkout to a fine-tuned French ASR model.

## Prerequisites

- NVIDIA GPU with ≥8 GB VRAM, CUDA 12.x driver
- ~80 GB free disk for raw + processed Common Voice
- `ffmpeg`, `uv`, `ollama`

```bash
make setup           # uv sync + sanity-checks CUDA / ffmpeg / Ollama
```

## Phase 1 — Data

```bash
make data            # downloads CV-FR to /mnt/d/french-data, preprocesses to data/processed
```

Verify with `notebooks/01_explore_dataset.ipynb` — duration histogram, char distribution, speaker count.

## Phase 2 — Baseline (zero-shot)

```bash
make baseline        # appends rows to docs/results/wer_table.md
```

Runs `facebook/wav2vec2-large-xlsr-53-french` on the test split and records WER as the reference ceiling for our fine-tune to beat.

## Phase 3 — Sanity training

```bash
uv run python scripts/train.py \
    --config configs/train.yaml \
    --max-steps 500 --train-frac 0.05 --epochs 1
```

Watch `tensorboard --logdir runs/` — CTC loss must decrease monotonically and sample decodes at step 500 should produce real French words.

## Phase 4 — Full fine-tune

```bash
make train           # 5–10 epochs, multi-day wall-clock on RTX 2000 Ada
```

Best-by-dev-WER checkpoint is auto-exported to `models/final/`.

## Phase 5 — Evaluation

```bash
make eval            # writes loss_curves.png + updates wer_table.md
```

**Goal**: test WER ≤ 20 %. If missed, escape hatches in priority order:

1. Train more epochs (`num_train_epochs: 15` in `configs/train.yaml`).
2. Add KenLM beam decoding: `uv sync --extra lm` then enable in `evaluate.py`.
3. Warm-start from `facebook/wav2vec2-large-xlsr-53-french` (set `model.backbone` in `configs/train.yaml`).

## Phase 6 — Tutor

```bash
make gradio          # browser UI at localhost:7860
make cli             # terminal push-to-talk
```
