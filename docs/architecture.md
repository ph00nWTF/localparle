# Architecture

## Pipeline

```
                ┌──────────────┐
   microphone ──▶│  audio_io.py │  16 kHz mono float32, VAD-trimmed
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │  ASREngine   │  models/final/ (wav2vec2-fr-xlsr fp16)
                └──────┬───────┘
                       │ transcript (str)
                       ▼
                ┌──────────────┐
                │  LLMEngine   │  Ollama /api/chat → mistral:instruct
                └──────┬───────┘
                       │ response_text (str)
                       ▼
                ┌──────────────┐
                │  TTSEngine   │  Piper (fr_FR-siwis-medium) → wav bytes
                └──────┬───────┘
                       ▼
                  loudspeaker
```

## Components

- **`src/french_tutor/audio_io.py`** — `record_until_silence()`, `play_wav_bytes()`, `list_devices()`.
- **`src/french_tutor/tutor.py`** — `ConversationPipeline.turn(audio) -> TurnResult`. Three engines (`ASREngine`, `LLMEngine`, `TTSEngine`) each with one public method; no inheritance.
- **`src/french_tutor/apps.py`** — `run_cli()` and `run_gradio()`. Both build the same `ConversationPipeline`.
- **`src/french_tutor/data.py` / `model.py` / `train.py`** — training-time only. Not loaded by the tutor app.

## Why this shape

- One file per concept; one public function/method per concept. Aligns with the project's simplicity-first guideline.
- `ConversationPipeline` is the only thing the apps know. Adding a Whisper backend or a Coqui TTS backend later is a matter of swapping one engine, not rewriting the app layer.

## VRAM budget on 8 GB

| Component | VRAM (approx) |
|---|---|
| Wav2vec2 (fp16, inference) | ~1.2 GB |
| Mistral 7B Q4_K_M (Ollama) | ~4.5 GB |
| CUDA context + workspace | ~0.8 GB |
| Headroom | ~1.5 GB |

Keep gradio's audio components on CPU; never hold both training + inference paths in memory at the same time.
