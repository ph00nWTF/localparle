# LocalParle

Offline French conversation tutor. You speak French into your microphone, the system writes down what you said, replies in French through a local chatbot, and speaks the response back. Nothing leaves your machine.

```
mic → fine-tuned wav2vec2 ASR → Mistral-Nemo (Ollama) → Piper TTS → speakers
```

Trained on Mozilla Common Voice French 25.0. Inference runs entirely on a single laptop with a modest NVIDIA GPU. No cloud, no API keys.

## Status

Complete. All four primary project goals are met. The Raspberry Pi stretch goal was not implemented.

## Headline results

Evaluated on the official Common Voice French test split (16,201 samples) using `jiwer`.

| Model | Decode | WER | CER |
|---|---|---|---|
| `wav2vec2-large-xlsr-53-french` (zero-shot reference) | greedy | 27.50% | n/a |
| **Ours (fine-tuned)** | greedy | **17.84%** | 5.18% |
| **Ours (fine-tuned)** | beam=100, KenLM 4-gram | **14.86%** | 4.47% |

Comparison chart at `docs/results/wer_cer_comparison.png`. Loss curve at `docs/results/loss_curves.png`. Full breakdown in `docs/results/wer_table.md`.

## Quickstart

```bash
make setup       # uv sync, verify CUDA, ffmpeg, Ollama
make data        # download and preprocess Common Voice FR
make baseline    # zero-shot WER reference (xls-r-53-french)
make train       # fine-tune wav2vec2 on Common Voice
make eval        # WER on the test split, optionally with KenLM
make serve       # browser-based tutor at http://localhost:8000
make cli         # terminal push-to-talk version
```

`make train` is a multi-day run on a single consumer GPU. The actual fine-tune in `models/final-v2/` was produced on 4x NVIDIA L40 GPUs in roughly 17 hours of active training time across 4 SLURM jobs.

## Pedagogy

What makes this a tutor and not just a chatbot is the prompt-building layer in `src/french_tutor/prompt_builder.py`. Every turn rebuilds Mistral's instructions from scratch based on:

- the learner's CEFR level (A1 through C1)
- the current scenario (topic, setting, opening line)
- vocabulary cards due for review under an SM-2 spaced-repetition schedule
- mode (guided opener or free-conversation sandbox)

Correction defaults to elicitation rather than recasts, following Lyster and Ranta 1997. Vocabulary i+1 difficulty follows Krashen 1982. Spaced repetition follows Woźniak 1987 (the original SuperMemo paper). Full citations are in `docs/tutor_pedagogy.md`.

There is also an English-fallback ASR. A1 learners often answer in English when they don't know how to respond. The system runs `faster-whisper-tiny` in parallel as a language detector. When English is dominant, the reply switches to English plus one short French phrase to mimic; otherwise the English text is passed as auxiliary context for code-switched speech.

## Project layout

```
configs/      data.yaml, train.yaml, train_v2.yaml, tutor.yaml, ...
data/         raw, processed, splits, curriculum, learner   (gitignored)
docs/         architecture.md, training.md, tutor_pedagogy.md,
              progress_report.md, wsl_setup.md, results/
models/       final-v2/, lm/, piper/                        (gitignored)
notebooks/    01_explore_dataset.ipynb, 02_error_analysis.ipynb
scripts/      prepare_data.py, train_v2.py, evaluate.py, tune_lm.py,
              build_lm.py, build_cefr_lookup.py, plot_wer_cer.py,
              tutor_cli.py, tutor_gradio.py, tutor_text.py, serve_web.py
src/french_tutor/
              tutor.py, prompt_builder.py, pedagogy.py, learner_state.py,
              audio_io.py, data_v2.py, model_v2.py, train_v2.py, ...
static/       index.html (browser UI)
tests/        test_data, test_metrics, test_tutor, test_vad,
              test_pedagogy, test_prompt_builder, test_learner_state, ...
```

## Hardware

- **Training**: 4x NVIDIA L40 on the KU HPC cluster. 10 epochs, 47,930 steps, effective batch size 128, about 17 hours of active GPU time across 4 SLURM jobs.
- **Inference**: any laptop with a modern NVIDIA GPU. ASR plus Piper TTS uses about 1.5 GB VRAM. Mistral-Nemo through Ollama adds roughly 6 GB. CPU-only inference works but is several times slower per turn.

## Citations

- Baevski et al. 2020, *wav2vec 2.0*, NeurIPS
- Conneau et al. 2021, *XLS-R*, Interspeech
- Ardila et al. 2020, *Common Voice*, LREC
- Heafield 2011, *KenLM*, WMT
- Krashen 1982, *Principles and Practice in Second Language Acquisition*, Pergamon
- Lyster & Ranta 1997, *Corrective Feedback and Learner Uptake*, SSLA 19(1)
- Woźniak 1987, *SuperMemo 2 Algorithm*

## License

MIT. See `LICENSE`.
