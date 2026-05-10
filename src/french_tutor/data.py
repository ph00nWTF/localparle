"""CV split readers, transcript normalization, vocab, CTC collator.

Audio source is the Mozilla Data Collective Common Voice tarball (mp3 + TSVs).
The tarball is downloaded by `scripts/prepare_data.py`; this module just reads
the extracted layout and hands HF `Dataset` objects downstream.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, Dataset

# Letters, spaces, apostrophes, hyphens — everything else stripped.
_TRANSCRIPT_KEEP = re.compile(r"[^a-zàâäéèêëîïôöùûüÿçñæœ '\-]")


def normalize_transcript(text: str) -> str:
    """NFC-normalize, lowercase, strip everything except letters, spaces, apostrophes, hyphens.

    Used for both training labels and the CTC vocab — keep them in sync.
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = _TRANSCRIPT_KEEP.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_cv_root(extracted_dir: Path, language: str = "fr") -> Path:
    """Locate the `cv-corpus-*/<language>/` directory inside an extracted CV tarball."""
    matches = sorted(extracted_dir.glob(f"cv-corpus-*/{language}"))
    if not matches:
        raise FileNotFoundError(
            f"No cv-corpus-*/{language}/ directory under {extracted_dir}. "
            "Did the tarball extract correctly?"
        )
    return matches[-1]


def read_cv_split_tsv(tsv_path: Path) -> list[dict[str, Any]]:
    """Read a CV split TSV (train.tsv / dev.tsv / test.tsv) into a list of row dicts."""
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_clip_durations(tsv_path: Path) -> dict[str, float]:
    """Read CV's `clip_durations.tsv` (columns: clip, duration[ms]) into clip → seconds.

    Using this lookup avoids decoding every mp3 just to learn its length — the 191k-clip
    decode pass over the WSL 9P bridge is what made the previous flow take hours.
    """
    out: dict[str, float] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out[row["clip"]] = int(row["duration[ms]"]) / 1000.0
    return out


def build_split_dataset(cv_root: Path, tsv_name: str, sample_rate: int) -> Dataset:
    """Construct an HF Dataset from a CV split TSV with a lazy-decoded `audio` column.

    `duration_s` is injected from `clip_durations.tsv` so downstream filters don't have to
    decode audio. The `audio` column is a path string that HF resolves to a resampled
    waveform on access.
    """
    rows = read_cv_split_tsv(cv_root / tsv_name)
    durations = read_clip_durations(cv_root / "clip_durations.tsv")
    clips_dir = cv_root / "clips"
    records = [
        {
            "client_id": r["client_id"],
            "audio": str(clips_dir / r["path"]),
            "sentence": r["sentence"],
            "up_votes": int(r.get("up_votes") or 0),
            "down_votes": int(r.get("down_votes") or 0),
            "duration_s": durations[r["path"]],
        }
        for r in rows
    ]
    ds = Dataset.from_list(records)
    return ds.cast_column("audio", Audio(sampling_rate=sample_rate))


def filter_dataset(
    ds: Dataset,
    *,
    min_duration_s: float,
    max_duration_s: float,
    drop_if_down_votes_exceed_up_votes: bool,
) -> Dataset:
    """Filter rows by `duration_s` and (optionally) up/down vote ratio.

    Uses `input_columns=` so HF only reads metadata — the lazy `audio` column is never
    decoded during filtering.
    """

    def _keep(duration_s: float, up_votes: int, down_votes: int) -> bool:
        in_duration = min_duration_s <= duration_s <= max_duration_s
        ok_votes = (not drop_if_down_votes_exceed_up_votes) or down_votes <= up_votes
        return in_duration and ok_votes

    return ds.filter(_keep, input_columns=["duration_s", "up_votes", "down_votes"])


def normalize_split(ds: Dataset) -> Dataset:
    """Apply normalize_transcript to the `sentence` column. Audio column is not read."""
    return ds.map(
        lambda sentence: {"sentence": normalize_transcript(sentence)},
        input_columns=["sentence"],
    )


def build_vocab(
    transcripts: list[str],
    *,
    unk: str,
    pad: str,
    word_delim: str,
) -> dict[str, int]:
    """Build a character-level CTC vocab from training transcripts only.

    The space character is replaced by `word_delim` so the tokenizer can use spaces in IDs.
    `pad` and `unk` are appended last (their IDs matter for CTC and the processor).
    """
    chars: set[str] = set()
    for t in transcripts:
        chars.update(t)
    chars.discard(" ")
    vocab_chars = sorted(chars)
    vocab: dict[str, int] = {ch: i for i, ch in enumerate(vocab_chars)}
    vocab[word_delim] = len(vocab)
    vocab[unk] = len(vocab)
    vocab[pad] = len(vocab)
    return vocab


@dataclass
class DataCollatorCTCWithPadding:
    """Pad input audio + labels for CTC training.

    Adapted from the official Hugging Face wav2vec2 fine-tuning recipe:
    https://huggingface.co/blog/fine-tune-wav2vec2-english
    """

    processor: Any
    padding: bool | str = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features, padding=self.padding, return_tensors="pt"
            )

        # CTC needs -100 in pad positions so they're ignored by the loss.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
