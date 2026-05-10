"""Grid search over pyctcdecode alpha/beta on 2000 dev-split samples.

Usage:
    uv run python scripts/tune_lm.py

Prints a WER table and the best alpha/beta to paste into configs/tutor.yaml.
Runs in ~12 min on GPU (25 combos × ~30 s each).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from jiwer import wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ── config ────────────────────────────────────────────────────────────────────
CHECKPOINT   = Path("models/final-v2")
LM_PATH      = Path("models/lm/lm.binary")
UNIGRAMS     = Path("models/lm/unigrams.txt")
DEV_PATH     = Path("data/processed/dev")
N_SAMPLES    = 2000   # subset of dev — enough for reliable ranking, fast to run
ALPHAS       = [0.2, 0.5, 0.8, 1.0, 1.5]
BETAS        = [0.5, 1.0, 1.5, 2.0, 2.5]
BEAM_WIDTH   = 100
SAMPLE_RATE  = 16000
# ──────────────────────────────────────────────────────────────────────────────

import io
import librosa
import torch


def _load_audio(audio_bytes: bytes) -> np.ndarray:
    wave, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    return wave.astype(np.float32)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model…")
    processor = Wav2Vec2Processor.from_pretrained(str(CHECKPOINT))
    model = Wav2Vec2ForCTC.from_pretrained(str(CHECKPOINT), use_safetensors=True).to(device).eval()

    print(f"Loading dev split (first {N_SAMPLES} samples)…")
    ds = load_from_disk(str(DEV_PATH)).select(range(N_SAMPLES))

    print("Computing logits (once)…")
    audio_col = ds._data.column("audio")
    sentence_col = ds._data.column("sentence")
    all_logits, all_refs = [], []
    for i in range(len(ds)):
        wave = _load_audio(audio_col[i].as_py()["bytes"])
        ref = sentence_col[i].as_py().strip().lower()
        inputs = processor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(
                inputs.input_values.to(device),
                attention_mask=inputs.attention_mask.to(device),
            ).logits
        all_logits.append(logits.cpu().numpy()[0])
        all_refs.append(ref)
    print(f"Logits ready for {len(all_logits)} samples.\n")

    from pyctcdecode import build_ctcdecoder
    vocab_dict = processor.tokenizer.get_vocab()
    labels = [t.lower() for t, _ in sorted(vocab_dict.items(), key=lambda kv: kv[1])]
    unigrams = UNIGRAMS.read_text(encoding="utf-8").splitlines()

    results = []
    for alpha in ALPHAS:
        for beta in BETAS:
            decoder = build_ctcdecoder(labels, str(LM_PATH), unigrams=unigrams,
                                       alpha=alpha, beta=beta)
            hyps = [decoder.decode(logits, beam_width=BEAM_WIDTH) for logits in all_logits]
            wer_val = wer(all_refs, hyps)
            results.append((wer_val, alpha, beta))
            print(f"  alpha={alpha:.1f}  beta={beta:.1f}  WER={wer_val:.4f}")

    results.sort()
    best_wer, best_alpha, best_beta = results[0]
    print(f"\nBest: alpha={best_alpha}  beta={best_beta}  WER={best_wer:.4f}")
    print(f"\nPaste into configs/tutor.yaml:")
    print(f"  alpha: {best_alpha}")
    print(f"  beta:  {best_beta}")

    out = Path("docs/results/lm_tune.json")
    out.write_text(json.dumps({"results": results, "best": {"alpha": best_alpha, "beta": best_beta, "wer": best_wer}}, indent=2))
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
