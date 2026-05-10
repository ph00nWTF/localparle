"""Evaluate the fine-tuned model on the test split — greedy CTC or beam+LM.

Greedy (no LM):
    uv run python scripts/evaluate.py --model-dir .../final-v2

Beam search + KenLM:
    uv run python scripts/evaluate.py --model-dir .../final-v2 \
        --lm-path /kuhpc/scratch/deng/e394n539/french-data/lm/fr_4gram.bin \
        --alpha 0.5 --beta 1.5 --beam-width 100

Computes WER, CER, and per-bucket WER (by reference length). Writes
`eval_results[_lm].json` next to the model checkpoint.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import time
from pathlib import Path

import librosa
import torch
from datasets import load_from_disk
from jiwer import cer, wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from french_tutor.config import load_yaml
from french_tutor.data import normalize_transcript
from french_tutor.utils import setup_logging

log = logging.getLogger("evaluate")


def _bucket_wer(refs: list[str], hyps: list[str]) -> dict[str, dict]:
    buckets = {"≤5 words": [], "6-10 words": [], "11-20 words": [], ">20 words": []}
    for r, h in zip(refs, hyps):
        n = len(r.split())
        if n <= 5:
            key = "≤5 words"
        elif n <= 10:
            key = "6-10 words"
        elif n <= 20:
            key = "11-20 words"
        else:
            key = ">20 words"
        buckets[key].append((r, h))
    out = {}
    for k, pairs in buckets.items():
        if not pairs:
            out[k] = {"count": 0, "wer": None}
            continue
        rs = [p[0] for p in pairs]
        hs = [p[1] for p in pairs]
        out[k] = {"count": len(pairs), "wer": float(wer(rs, hs))}
    return out


def _build_lm_decoder(processor, lm_path: Path, alpha: float, beta: float):
    """Build a pyctcdecode BeamSearchDecoderCTC with the given KenLM model."""
    from pyctcdecode import build_ctcdecoder

    vocab_dict = processor.tokenizer.get_vocab()
    labels = [None] * len(vocab_dict)
    for tok, idx in vocab_dict.items():
        labels[idx] = tok

    # pyctcdecode expects "" for the CTC blank label
    labels = [t if t != processor.tokenizer.pad_token else "" for t in labels]

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=str(lm_path),
        alpha=alpha,
        beta=beta,
    )
    return decoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/train_v2.yaml", type=Path)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--lm-path", default=None, type=Path,
                        help="If set, use pyctcdecode + KenLM beam search instead of greedy")
    parser.add_argument("--alpha", default=0.5, type=float, help="LM weight (with --lm-path)")
    parser.add_argument("--beta", default=1.5, type=float, help="length bonus (with --lm-path)")
    parser.add_argument("--beam-width", default=100, type=int, help="beam width (with --lm-path)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_yaml(args.config)
    processed_dir = Path(cfg["data"]["processed_dir"])
    test_path = processed_dir / "test"
    use_lm = args.lm_path is not None

    log.info("loading model from %s", args.model_dir)
    processor = Wav2Vec2Processor.from_pretrained(str(args.model_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(args.model_dir), use_safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    log.info("model on %s", device)

    decoder = None
    if use_lm:
        log.info("building beam-search decoder: lm=%s alpha=%.2f beta=%.2f beam=%d",
                 args.lm_path, args.alpha, args.beta, args.beam_width)
        decoder = _build_lm_decoder(processor, args.lm_path, args.alpha, args.beta)

    log.info("loading test split from %s", test_path)
    test_ds = load_from_disk(str(test_path))
    if args.max_samples is not None:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))
    log.info("evaluating %d samples (mode=%s)", len(test_ds), "beam+LM" if use_lm else "greedy")

    sentences_all = test_ds["sentence"]
    raw_audio = test_ds._data["audio"].to_pylist()

    references: list[str] = []
    hypotheses: list[str] = []

    start = time.time()
    with torch.no_grad():
        for i in range(0, len(test_ds), args.batch_size):
            batch_raw = raw_audio[i : i + args.batch_size]
            arrays = [
                librosa.load(
                    io.BytesIO(item["bytes"]) if item["bytes"] else item["path"],
                    sr=16000, mono=True,
                )[0]
                for item in batch_raw
            ]
            inputs = processor(arrays, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)
            logits = model(input_values, attention_mask=attention_mask).logits

            if use_lm:
                logits_np = logits.cpu().numpy()
                preds = [decoder.decode(logits_np[j], beam_width=args.beam_width)
                         for j in range(logits_np.shape[0])]
            else:
                predicted_ids = torch.argmax(logits, dim=-1)
                preds = processor.batch_decode(predicted_ids)

            hypotheses.extend(normalize_transcript(p) for p in preds)
            references.extend(normalize_transcript(s) for s in sentences_all[i : i + args.batch_size])
            if (i // args.batch_size + 1) % 20 == 0:
                done = min(i + args.batch_size, len(test_ds))
                elapsed = time.time() - start
                rate = done / elapsed
                eta = (len(test_ds) - done) / rate if rate > 0 else 0
                log.info("  %d / %d (%.1f s/s, eta %.0fs)", done, len(test_ds), rate, eta)

    elapsed = time.time() - start
    overall_wer = float(wer(references, hypotheses))
    overall_cer = float(cer(references, hypotheses))
    buckets = _bucket_wer(references, hypotheses)

    print("\n=== Evaluation results ===")
    print(f"Model:    {args.model_dir}")
    print(f"Decode:   {'beam+LM (alpha=' + str(args.alpha) + ', beta=' + str(args.beta) + ', beam=' + str(args.beam_width) + ')' if use_lm else 'greedy'}")
    if use_lm:
        print(f"LM:       {args.lm_path}")
    print(f"Test set: {len(test_ds)} samples")
    print(f"Elapsed:  {elapsed:.1f}s ({len(test_ds)/elapsed:.1f} samples/sec)")
    print(f"WER:      {overall_wer:.4f}  ({overall_wer*100:.2f}%)")
    print(f"CER:      {overall_cer:.4f}  ({overall_cer*100:.2f}%)")
    print("\nWER by reference length:")
    for k, v in buckets.items():
        if v["count"] == 0:
            print(f"  {k:<14} —")
        else:
            print(f"  {k:<14} n={v['count']:5d}  WER={v['wer']*100:.2f}%")

    sample_pairs = [{"ref": r, "hyp": h} for r, h in zip(references[:10], hypotheses[:10])]

    suffix = "_lm" if use_lm else ""
    out_path = args.model_dir / f"eval_results{suffix}.json"
    out_path.write_text(json.dumps({
        "model_dir": str(args.model_dir),
        "decode": "beam+LM" if use_lm else "greedy",
        "lm_path": str(args.lm_path) if use_lm else None,
        "alpha": args.alpha if use_lm else None,
        "beta": args.beta if use_lm else None,
        "beam_width": args.beam_width if use_lm else None,
        "n_samples": len(test_ds),
        "elapsed_sec": elapsed,
        "wer": overall_wer,
        "cer": overall_cer,
        "buckets": buckets,
        "samples": sample_pairs,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
