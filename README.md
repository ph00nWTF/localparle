# French Tutor

Local AI French language tutor. Fine-tunes **wav2vec 2.0 (XLS-R 300M)** on the **Mozilla Common Voice French** corpus, then drives a real-time spoken conversation through:

```
mic → fine-tuned ASR → Mistral 7B (Ollama) → Piper TTS → speakers
```

Runs entirely on a single laptop with an NVIDIA RTX 2000 Ada (8 GB VRAM). No cloud, no OpenAI.

## Status

Bootstrap. See [`french_proposal.md`](french_proposal.md) for the project proposal and [`rubric.md`](rubric.md) for the self-assessment rubric.

## Headline results

| Model | Test WER | Notes |
|---|---|---|
| `facebook/wav2vec2-large-xlsr-53-french` (public) | 27.5 % | Phase 2 reference |
| Ours (fine-tuned on CV-FR) | TBD | Phase 5 target: ≤ 20 % |

Loss curves: `docs/results/loss_curves.png`. Full table: `docs/results/wer_table.md`.

## Quickstart

```bash
make setup       # uv sync + verify CUDA / ffmpeg / Ollama
make data        # download + preprocess Common Voice FR (~80 GB on /mnt/d)
make baseline    # WER reference number (facebook/wav2vec2-large-xlsr-53-french)
make train       # full fine-tune (multi-day on RTX 2000 Ada)
make eval        # WER on test split, generate plots/tables
make gradio      # interactive tutor at http://localhost:7860
```

CLI version (also what runs on the Pi): `make cli`.

## Project layout

```
configs/      data.yaml, train.yaml, tutor.yaml
data/         raw -> /mnt/d/french-data/raw  (gitignored)
              processed/  splits/  vocab.json  (gitignored)
docs/         architecture.md, training.md, pi-deployment.md, results/
models/       checkpoints/  final/  (gitignored)
notebooks/    01_explore_dataset.ipynb, 02_error_analysis.ipynb
scripts/      prepare_data.py, train.py, evaluate.py, tutor_cli.py, tutor_gradio.py
src/french_tutor/
              data.py, model.py, train.py, tutor.py, audio_io.py, apps.py, ...
tests/        test_data.py, test_metrics.py, test_tutor.py, fixtures/
```

## Hardware

- GPU: NVIDIA RTX 2000 Ada Generation (8 GB)
- Disk: raw CV-FR on `/mnt/d/french-data/` (~60 GB), processed on WSL ext4 (~30 GB hot)

## License

MIT — see [`LICENSE`](LICENSE).
