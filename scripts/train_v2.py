"""Fine-tune wav2vec2-XLS-R on Common Voice French — DDP-correct rewrite.

Run:
    uv run torchrun --nproc_per_node=N scripts/train_v2.py --config configs/train_v2.yaml

Design (vs. v1):
- Sentinel files for preprocessing rank sync (dist not yet initialized at this point)
- Timeout on the wait loop (v1 hung forever if rank 0 crashed)
- Only rank 0 loads raw split (v1 wasted memory on every rank)
- freeze_feature_encoder() — current API name
- processor.tokenizer.pad() — replaces deprecated as_target_processor()
- processing_class= in Trainer — replaces deprecated tokenizer=
- compute_metrics does not mutate inputs (v1 mutated label_ids in place)
- fsync after sentinel write — ensures shared filesystem propagation
- gradient_checkpointing must be false (validated; DDP + frozen params + checkpointing fails)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from french_tutor.config import load_yaml
from french_tutor.utils import set_seed, setup_logging

log = logging.getLogger("train_v2")


def _is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def _fsync_path(path: Path) -> None:
    """Force flush of a file's directory entry to disk so other ranks can see it."""
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _preprocess_on_rank0(
    raw_split_path: Path,
    cache_path: Path,
    processor,
    train_frac: float,
) -> None:
    """Rank 0 only: load raw split, optionally subsample, preprocess to cache.

    Idempotent: if sentinel exists, skip. If cache_path exists without sentinel
    (partial cache from a crashed run), remove it and start over.
    """
    import io
    import shutil

    import librosa
    from datasets import Dataset, load_from_disk

    sentinel = cache_path.parent / f".{cache_path.name}_ok"
    if sentinel.exists():
        log.info("preprocessing cache already complete: %s", cache_path)
        return

    if cache_path.exists():
        log.warning("partial cache found without sentinel; removing %s", cache_path)
        shutil.rmtree(cache_path)

    log.info("loading raw split from %s", raw_split_path)
    raw = load_from_disk(str(raw_split_path))

    if 0.0 < train_frac < 1.0:
        n_sub = max(1, int(len(raw) * train_frac))
        raw = raw.select(range(n_sub))
        log.info("subsampled to %d examples (frac=%.4f)", n_sub, train_frac)

    audio_col = raw._data["audio"]
    sentences = raw["sentence"]
    n = len(raw)
    log.info("preprocessing %d examples", n)

    def gen():
        for i in range(n):
            item = audio_col[i].as_py()
            array = librosa.load(io.BytesIO(item["bytes"]), sr=16000, mono=True)[0]
            input_values = processor(array, sampling_rate=16000).input_values[0]
            labels = processor.tokenizer(sentences[i]).input_ids
            if (i + 1) % 10000 == 0:
                log.info("    %d / %d", i + 1, n)
            yield {"input_values": input_values, "labels": labels}

    ds = Dataset.from_generator(gen)
    ds.save_to_disk(str(cache_path))
    sentinel.touch()
    _fsync_path(sentinel)
    log.info("preprocessing complete: %s", cache_path)


def _wait_for_cache_with_timeout(cache_path: Path, timeout_s: int = 7200) -> None:
    """Non-rank-0: wait up to timeout_s seconds for rank 0 to finish preprocessing."""
    import time

    sentinel = cache_path.parent / f".{cache_path.name}_ok"
    rank = os.environ.get("LOCAL_RANK", "?")
    log.info("rank %s waiting for sentinel %s", rank, sentinel)
    start = time.time()
    while not sentinel.exists():
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise RuntimeError(
                f"rank {rank}: preprocessing sentinel did not appear "
                f"within {timeout_s}s — rank 0 likely crashed"
            )
        time.sleep(10)
    log.info("rank %s: sentinel found after %.1fs", rank, time.time() - start)


def _validate_paths(processed_dir: Path, vocab_path: Path) -> None:
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir does not exist: {processed_dir}")
    if not (processed_dir / "train").exists():
        raise FileNotFoundError(f"train split missing: {processed_dir / 'train'}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab file missing: {vocab_path}")


def _resolve_train_frac(value) -> float:
    if value is None:
        return 1.0
    f = float(value)
    if not (0.0 < f <= 1.0):
        raise ValueError(f"train_frac must be in (0, 1], got {f}")
    return f


def _run_finetune(cfg: dict) -> int:
    from datasets import load_from_disk

    from french_tutor.model_v2 import build_model, build_processor
    from french_tutor.train_v2 import build_trainer

    processed_dir = Path(cfg["data"]["processed_dir"])
    vocab_path = Path("data/vocab.json")
    train_frac = _resolve_train_frac(cfg["sampling"].get("train_frac"))

    _validate_paths(processed_dir, vocab_path)

    log.info("building processor from %s", vocab_path)
    processor = build_processor(vocab_path, sample_rate=16000)

    log.info("building model from %s", cfg["model"]["backbone"])
    model = build_model(cfg["model"]["backbone"], processor, cfg["model"])

    frac_tag = f"_frac{int(train_frac * 1000):04d}" if train_frac < 1.0 else ""
    train_cache = processed_dir / f"preprocessed_train{frac_tag}"
    raw_train_path = processed_dir / "train"

    if _is_main_process():
        _preprocess_on_rank0(raw_train_path, train_cache, processor, train_frac)
    else:
        _wait_for_cache_with_timeout(train_cache)

    log.info("rank %s loading preprocessed train", os.environ.get("LOCAL_RANK", 0))
    train_ds = load_from_disk(str(train_cache))

    eval_ds = None
    if cfg["training"]["eval_strategy"] != "no":
        eval_cache = processed_dir / "preprocessed_dev"
        raw_dev_path = processed_dir / "dev"
        if _is_main_process():
            _preprocess_on_rank0(raw_dev_path, eval_cache, processor, 1.0)
        else:
            _wait_for_cache_with_timeout(eval_cache)
        eval_ds = load_from_disk(str(eval_cache))

    if _is_main_process():
        log.info(
            "train=%d  eval=%s",
            len(train_ds),
            len(eval_ds) if eval_ds is not None else "skipped",
        )

    trainer = build_trainer(model, processor, train_ds, eval_ds, cfg["training"], cfg["sampling"])

    output_dir = Path(cfg["training"]["output_dir"])
    has_checkpoint = output_dir.exists() and any(output_dir.glob("checkpoint-*"))

    trainer.train(resume_from_checkpoint=True if has_checkpoint else None)

    if _is_main_process():
        final_dir = Path(cfg["export"]["final_dir"])
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        processor.save_pretrained(str(final_dir))
        log.info("saved final model + processor to %s", final_dir)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_v2.yaml", type=Path)
    args = parser.parse_args()

    setup_logging()
    cfg = load_yaml(args.config)
    set_seed(cfg["training"]["seed"])
    return _run_finetune(cfg)


if __name__ == "__main__":
    sys.exit(main())
