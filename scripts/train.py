"""Fine-tune wav2vec2-XLS-R on Common Voice French (or run zero-shot baseline).

    uv run python scripts/train.py --config configs/train.yaml
    uv run python scripts/train.py --config configs/train.yaml --baseline
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from french_tutor.config import load_yaml
from french_tutor.utils import set_seed, setup_logging

log = logging.getLogger("train")

_BASELINE_MODEL = "facebook/wav2vec2-large-xlsr-53-french"


def _run_baseline(cfg: dict) -> int:
    import io

    import librosa
    import torch
    from datasets import load_from_disk
    from jiwer import wer as compute_wer
    from torch.utils.tensorboard import SummaryWriter
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    from french_tutor.data import normalize_transcript
    from french_tutor.model import greedy_decode

    processed_dir = Path(cfg["data"]["processed_dir"])
    log.info("loading test split from %s", processed_dir / "test")
    test_ds = load_from_disk(str(processed_dir / "test"))
    log.info("test split: %d examples", len(test_ds))

    log.info("loading baseline model: %s", _BASELINE_MODEL)
    processor = Wav2Vec2Processor.from_pretrained(_BASELINE_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(_BASELINE_MODEL, use_safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    log.info("model on %s", device)

    # datasets 4.x requires torchcodec for Audio decode; bypass by reading raw paths
    # from the Arrow table and loading mp3s with librosa (already a project dep).
    sentences_all = test_ds["sentence"]
    raw_audio = test_ds._data["audio"].to_pylist()

    batch_size = cfg["training"]["per_device_eval_batch_size"]
    references: list[str] = []
    hypotheses: list[str] = []

    with torch.no_grad():
        for i in range(0, len(test_ds), batch_size):
            batch_raw = raw_audio[i : i + batch_size]
            arrays = [
                librosa.load(io.BytesIO(item["bytes"]) if item["bytes"] else item["path"], sr=16000, mono=True)[0]
                for item in batch_raw
            ]
            inputs = processor(
                arrays,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)
            logits = model(input_values, attention_mask=attention_mask).logits
            preds = greedy_decode(logits, processor)
            hypotheses.extend(normalize_transcript(p) for p in preds)
            references.extend(sentences_all[i : i + batch_size])
            if (i // batch_size + 1) % 20 == 0:
                log.info("  %d / %d", min(i + batch_size, len(test_ds)), len(test_ds))

    score = compute_wer(references, hypotheses)
    log.info("baseline WER: %.4f  (%.1f%%)", score, score * 100)

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output_dir / "baseline"))
    writer.add_scalar("eval/wer", score, global_step=0)
    writer.close()

    (output_dir / "baseline_wer.txt").write_text(
        f"model: {_BASELINE_MODEL}\nwer: {score:.4f}\n"
    )
    log.info("saved baseline_wer.txt to %s", output_dir)
    return 0


def _run_finetune(cfg: dict) -> int:
    import io

    import librosa
    from datasets import Dataset, load_from_disk

    from french_tutor.model import build_model, build_processor
    from french_tutor.train import build_trainer

    processed_dir = Path(cfg["data"]["processed_dir"])
    vocab_path = Path("data/vocab.json")

    log.info("building processor from %s", vocab_path)
    processor = build_processor(vocab_path, sample_rate=16000)

    log.info("building model from %s", cfg["model"]["backbone"])
    model = build_model(cfg["model"]["backbone"], processor, cfg["model"])

    def _stream_preprocess(ds, cache_path: Path) -> Dataset:
        """Stream-decode mp3 → input_values + labels, saving to an explicit cache path.

        Uses a sentinel file to signal completion so DDP ranks can safely wait on rank 0
        without relying on Dataset.from_generator's fingerprint (which is non-deterministic
        across torchrun processes and causes cache misses on every DDP run).
        """
        import time
        sentinel = cache_path.parent / f".{cache_path.name}_ok"

        if local_rank == 0 and not sentinel.exists():
            audio_col = ds._data["audio"]
            sentences = ds["sentence"]
            n = len(ds)

            def gen():
                for i in range(n):
                    item = audio_col[i].as_py()
                    array = librosa.load(io.BytesIO(item["bytes"]), sr=16000, mono=True)[0]
                    input_values = processor(array, sampling_rate=16000).input_values[0]
                    labels = processor.tokenizer(sentences[i]).input_ids
                    if (i + 1) % 10000 == 0:
                        log.info("    %d / %d", i + 1, n)
                    yield {"input_values": input_values, "labels": labels}

            result = Dataset.from_generator(gen)
            result.save_to_disk(str(cache_path))
            sentinel.touch()
            log.info("cached preprocessed data to %s", cache_path)
        elif local_rank != 0:
            while not sentinel.exists():
                time.sleep(5)

        return load_from_disk(str(cache_path))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    train_frac = cfg["sampling"]["train_frac"]
    frac_tag = f"_frac{int(train_frac * 1000):04d}" if train_frac < 1.0 else ""

    if local_rank == 0:
        log.info("loading train split")
        train_raw = load_from_disk(str(processed_dir / "train"))
        if train_frac < 1.0:
            n_sub = max(1, int(len(train_raw) * train_frac))
            train_raw = train_raw.select(range(n_sub))
            log.info("subsampled train to %d examples", n_sub)
        log.info("preprocessing train (streaming)")
    else:
        train_raw = load_from_disk(str(processed_dir / "train"))

    train_cache = processed_dir / f"preprocessed_train{frac_tag}"
    train_ds = _stream_preprocess(train_raw, train_cache)

    eval_ds = None
    if cfg["training"]["eval_strategy"] != "no":
        if local_rank == 0:
            log.info("preprocessing dev (streaming)")
            dev_raw = load_from_disk(str(processed_dir / "dev"))
        else:
            dev_raw = load_from_disk(str(processed_dir / "dev"))
        eval_cache = processed_dir / "preprocessed_dev"
        eval_ds = _stream_preprocess(dev_raw, eval_cache)

    if local_rank == 0:
        log.info(
            "train=%d  eval=%s",
            len(train_ds),
            len(eval_ds) if eval_ds is not None else "skipped (eval_strategy=no)",
        )
    trainer = build_trainer(model, processor, train_ds, eval_ds, cfg["training"], cfg["sampling"])
    output_dir = Path(cfg["training"]["output_dir"])
    checkpoint = output_dir if any(output_dir.glob("checkpoint-*")) else None
    trainer.train(resume_from_checkpoint=checkpoint)

    final_dir = Path(cfg["export"]["final_dir"])
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    log.info("saved model + processor to %s", final_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml", type=Path)
    parser.add_argument("--baseline", action="store_true", help="Zero-shot WER, no fine-tuning")
    args = parser.parse_args()

    setup_logging()
    cfg = load_yaml(args.config)
    set_seed(cfg["training"]["seed"])

    if args.baseline:
        return _run_baseline(cfg)
    return _run_finetune(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
