"""Download + preprocess Mozilla Common Voice French via Mozilla Data Collective.

    uv run python scripts/prepare_data.py --config configs/data.yaml

Before running:
    1. Sign up at https://mozilladatacollective.com/auth/signup
    2. On the dataset page for Common Voice (French), accept the terms.
    3. Generate an API key at https://mozilladatacollective.com/profile/credentials
    4. export MDC_API_KEY=...
    5. Paste the dataset id (the slug after `/datasets/` in the URL) into
       configs/data.yaml under `mdc.dataset_id`.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tarfile
from pathlib import Path

from datacollective import download_dataset

from french_tutor.config import load_yaml
from french_tutor.data import (
    build_split_dataset,
    build_vocab,
    filter_dataset,
    find_cv_root,
    normalize_split,
)
from french_tutor.utils import setup_logging

log = logging.getLogger("prepare_data")


def _check_mdc_setup(cfg: dict) -> str:
    if not os.environ.get("MDC_API_KEY"):
        log.error(
            "MDC_API_KEY not set. Run `export MDC_API_KEY=...` after generating one at "
            "https://mozilladatacollective.com/profile/credentials (see data/README.md)."
        )
        sys.exit(2)
    dataset_id = (cfg["mdc"].get("dataset_id") or "").strip()
    if not dataset_id:
        log.error(
            "configs/data.yaml: mdc.dataset_id is empty. Browse the MDC dataset page for "
            "Common Voice (French), accept the terms, and paste the trailing slug from "
            "the URL into mdc.dataset_id."
        )
        sys.exit(2)
    return dataset_id


def _extract_tarball(tarball_path: Path, dest: Path) -> None:
    """Extract `tarball_path` into `dest`, idempotent via a `.extracted_ok` marker.

    The marker is written only after `tar.extractall` returns. If the previous run was
    interrupted mid-extract, the marker is absent and we re-extract from scratch — a
    half-extracted tree is worse than no skip, since downstream code assumes every clip
    referenced by a TSV is on disk.
    """
    marker = dest / ".extracted_ok"
    if marker.exists():
        log.info("extraction marker present at %s, skipping extract", marker)
        return
    dest.mkdir(parents=True, exist_ok=True)
    log.info("extracting %s to %s (this can take a while)", tarball_path, dest)
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    marker.touch()
    log.info("extraction complete; wrote marker %s", marker)


def _write_split_tsv(rows: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["client_id", "duration_s", "sentence"])
        for row in rows:
            writer.writerow([row["client_id"], f"{row['duration_s']:.3f}", row["sentence"]])
    log.info("wrote %s (%d rows)", dest, len(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", type=Path)
    args = parser.parse_args()

    setup_logging()
    cfg = load_yaml(args.config)
    dataset_id = _check_mdc_setup(cfg)

    download_dir = Path(cfg["mdc"]["download_dir"])
    download_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading MDC dataset %s into %s (auto-resume)", dataset_id, download_dir)
    tarball_path = Path(download_dataset(dataset_id, download_directory=str(download_dir)))
    log.info("download complete: %s", tarball_path)

    extracted_dir = Path(cfg["mdc"]["extracted_dir"])
    _extract_tarball(tarball_path, extracted_dir)
    cv_root = find_cv_root(extracted_dir, language="fr")
    log.info("CV root: %s", cv_root)

    processed = {}
    counts = {}
    for split_name, tsv_name in [("train", "train.tsv"), ("dev", "dev.tsv"), ("test", "test.tsv")]:
        log.info("building split=%s from %s", split_name, tsv_name)
        s = build_split_dataset(cv_root, tsv_name, sample_rate=cfg["audio"]["sample_rate"])
        s = filter_dataset(
            s,
            min_duration_s=cfg["audio"]["min_duration_s"],
            max_duration_s=cfg["audio"]["max_duration_s"],
            drop_if_down_votes_exceed_up_votes=cfg["filtering"]["drop_if_down_votes_exceed_up_votes"],
        )
        s = normalize_split(s)
        processed[split_name] = s
        counts[split_name] = len(s)
        log.info("split=%s rows=%d", split_name, len(s))

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name, s in processed.items():
        out = processed_dir / split_name
        s.save_to_disk(str(out))
        log.info("saved processed split to %s", out)

    splits_dir = Path(cfg["paths"]["splits_dir"])
    for split_name, s in processed.items():
        # Column-only access — iterating Dataset rows would materialize the lazy `audio`
        # column and trigger the torchcodec decoder we're trying to avoid in this script.
        client_ids = s["client_id"]
        durations = s["duration_s"]
        sentences = s["sentence"]
        rows = [
            {"client_id": cid, "duration_s": d, "sentence": sent}
            for cid, d, sent in zip(client_ids, durations, sentences, strict=True)
        ]
        _write_split_tsv(rows, splits_dir / f"{split_name}.tsv")

    train_sentences = list(processed["train"]["sentence"])
    vocab = build_vocab(
        train_sentences,
        unk=cfg["vocab"]["unk_token"],
        pad=cfg["vocab"]["pad_token"],
        word_delim=cfg["vocab"]["word_delimiter"],
    )
    vocab_path = Path(cfg["paths"]["vocab_path"])
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    log.info("wrote vocab (%d tokens) to %s", len(vocab), vocab_path)

    log.info("done. counts: %s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
