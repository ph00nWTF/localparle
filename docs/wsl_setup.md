# WSL Setup — running the French Tutor locally

Reference: do these in order. Stop and fix at the first failing step.

---

## 1. Pull the latest code

```bash
cd ~/work/FrenchAI/french
git pull origin main
```

## 2. System packages (Ubuntu/Debian WSL)

```bash
sudo apt update
sudo apt install -y libportaudio2 libsndfile1 ffmpeg
```

- `libportaudio2` → required by `sounddevice` (mic capture).
- `libsndfile1` → required by `soundfile` (WAV decoding).
- `ffmpeg` → librosa decoder fallback for various audio formats.

## 3. Python deps via uv

If uv isn't installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
cd ~/work/FrenchAI/french
uv sync                    # installs runtime deps
uv sync --extra dev        # adds pytest etc. (optional)
```

Verify:
```bash
uv run python -c "import torch, transformers, pyctcdecode, kenlm, webrtcvad, sounddevice, soundfile, gradio; print('OK')"
```

## 4. Get the trained model + KenLM from HPC

Files needed locally:
- `models/final-v2/` (1.2 GB) — wav2vec2 fine-tuned weights + processor
- `models/lm/fr_4gram.arpa` (379 MB) — French n-gram LM

```bash
mkdir -p ~/work/FrenchAI/french/models/final-v2 ~/work/FrenchAI/french/models/lm

rsync -avz --progress \
  e394n539@<hpc-host>:/kuhpc/scratch/deng/e394n539/french-data/models/final-v2/ \
  ~/work/FrenchAI/french/models/final-v2/

rsync -avz --progress \
  e394n539@<hpc-host>:/kuhpc/scratch/deng/e394n539/french-data/lm/fr_4gram.arpa \
  ~/work/FrenchAI/french/models/lm/
```

Replace `<hpc-host>` with the HPC SSH alias you use. Total ~1.5 GB; expect 5-30 min.

## 5. Point the config at local paths

Edit `configs/tutor.yaml`:

```yaml
asr:
  checkpoint_dir: models/final-v2
  device: cuda          # or "cpu" if no GPU on WSL
  dtype: float16        # change to float32 if "cpu"
  lm_path: models/lm/fr_4gram.arpa
  alpha: 0.5
  beta: 1.5
  beam_width: 100
```

CPU notes: float16 on CPU breaks; switch to `float32` and expect ~2-5 sec per utterance.

## 6. Ollama + Mistral

Install Ollama: https://ollama.com/download. Then:

```bash
ollama serve &              # leave running in another shell or as a service
ollama pull mistral:instruct
ollama list                 # confirm "mistral:instruct" present
curl http://localhost:11434/api/tags  # confirm HTTP API responsive
```

## 7. Piper TTS + French voice

Install (release binaries are easiest):
```bash
mkdir -p ~/.local/piper && cd ~/.local/piper
wget https://github.com/rhasspy/piper/releases/latest/download/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz
ln -sf $HOME/.local/piper/piper/piper $HOME/.local/bin/piper
```

Download voice files (siwis = clear neutral voice):
```bash
mkdir -p ~/work/FrenchAI/french/voices && cd ~/work/FrenchAI/french/voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json
```

Test Piper:
```bash
echo "Bonjour, je suis votre tuteur." | piper --model voices/fr_FR-siwis-medium.onnx --output_file /tmp/test.wav
aplay /tmp/test.wav   # or play, mpv, ffplay
```

Then in `configs/tutor.yaml`:
```yaml
tts:
  voice: voices/fr_FR-siwis-medium.onnx
  binary_path: null   # uses "piper" from $PATH (the symlink we made)
```

## 8. Smoke test — non-interactive

Confirm the pipeline assembles without crashing:
```bash
uv run python -c "
from french_tutor.config import load_yaml
from french_tutor.tutor import build_pipeline
cfg = load_yaml('configs/tutor.yaml')
p = build_pipeline(cfg)
print('pipeline built OK')
"
```

This loads the model, processor, KenLM, and validates Mistral + Piper connectivity (LLM/TTS errors only show up when actually called).

## 9. Run the Gradio UI

```bash
uv run python scripts/tutor_gradio.py
```

Open http://127.0.0.1:7860. Click the mic, speak French ("Bonjour, comment allez-vous ?"), verify:
- Transcript appears in chat
- Tutor reply is reasonable French
- Reply audio plays automatically

If WSL doesn't expose mic to the browser by default, use `chrome.exe http://127.0.0.1:7860` from Windows side instead of WSL's browser.

## 10. Run the CLI (optional)

```bash
uv run python scripts/tutor_cli.py
```

Press Enter, speak, pause, hear reply. Type `reset` to clear conversation history, `quit` to exit.

---

## Common failures

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: webrtcvad` import — `pkg_resources` | `pip install "setuptools<81"` (already pinned in pyproject) |
| `OSError: PortAudio library not found` | `sudo apt install libportaudio2` |
| `FileNotFoundError: piper` | symlink piper into `~/.local/bin` (step 7) |
| `httpx.ConnectError: localhost:11434` | `ollama serve` not running, or wrong URL in config |
| Ollama replies in English | Mistral persona issue — check system_prompt_fr.txt is loading |
| Transcript is gibberish | Check `lm_path` points at .arpa, not .bin (auto-extracts unigrams from ARPA only) |
| Mic not picked up in Gradio | WSL2 needs PulseAudio bridge, or use Chrome on Windows side pointing at the WSL port |

---

## Performance expectations

- ASR (16-sec clip, GPU): ~0.5-1 sec
- ASR (16-sec clip, CPU): ~3-5 sec
- Mistral 7B reply: ~1-2 sec (after first warm-up call)
- Piper TTS (3-sentence reply): ~0.3-0.6 sec

Round-trip per turn: ~2-4 sec on GPU, ~5-10 sec on CPU.
