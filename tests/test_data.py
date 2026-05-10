"""Tests for src/french_tutor/data.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from french_tutor.data import (
    build_vocab,
    find_cv_root,
    normalize_transcript,
    read_clip_durations,
    read_cv_split_tsv,
)


def test_normalize_strips_punctuation_keeps_apostrophes_and_hyphens():
    assert normalize_transcript("Bonjour, ça va ?") == "bonjour ça va"
    assert normalize_transcript("C'est-à-dire !") == "c'est-à-dire"
    assert normalize_transcript("L'étoile, hier...") == "l'étoile hier"


def test_normalize_collapses_whitespace():
    assert normalize_transcript("  trop   d'espaces  ") == "trop d'espaces"


def test_normalize_lowercases_and_keeps_french_accents():
    assert normalize_transcript("Élève À ÊTRE") == "élève à être"


def test_normalize_strips_digits_and_emoji():
    assert normalize_transcript("J'ai 42 chats 🐱") == "j'ai chats"


def test_build_vocab_only_uses_training_chars():
    train = ["bonjour", "salut", "comment ça va"]
    vocab = build_vocab(train, unk="[UNK]", pad="[PAD]", word_delim="|")

    train_chars = set("".join(train)) - {" "}
    for ch in train_chars:
        assert ch in vocab

    # Special tokens come last and are distinct.
    assert vocab["|"] == len(train_chars)
    assert vocab["[UNK]"] == len(train_chars) + 1
    assert vocab["[PAD]"] == len(train_chars) + 2

    # Char never seen in training is not in vocab.
    assert "z" not in vocab


def test_build_vocab_is_deterministic():
    train = ["chat", "tac", "chat"]
    v1 = build_vocab(train, unk="[UNK]", pad="[PAD]", word_delim="|")
    v2 = build_vocab(list(reversed(train)), unk="[UNK]", pad="[PAD]", word_delim="|")
    assert v1 == v2


def test_read_cv_split_tsv_parses_rows(tmp_path: Path):
    tsv = tmp_path / "train.tsv"
    tsv.write_text(
        "client_id\tpath\tsentence\tup_votes\tdown_votes\n"
        "abc\tclip1.mp3\tBonjour\t2\t0\n"
        "def\tclip2.mp3\tÇa va\t1\t1\n",
        encoding="utf-8",
    )
    rows = read_cv_split_tsv(tsv)
    assert [r["client_id"] for r in rows] == ["abc", "def"]
    assert rows[1]["sentence"] == "Ça va"
    assert rows[0]["up_votes"] == "2"  # DictReader returns strs; build_split_dataset casts to int


def test_find_cv_root_locates_language_subdir(tmp_path: Path):
    cv_dir = tmp_path / "cv-corpus-99.0-2026-04-27" / "fr"
    cv_dir.mkdir(parents=True)
    assert find_cv_root(tmp_path, language="fr") == cv_dir


def test_find_cv_root_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_cv_root(tmp_path, language="fr")


def test_read_clip_durations_parses_ms_to_seconds(tmp_path: Path):
    tsv = tmp_path / "clip_durations.tsv"
    tsv.write_text(
        "clip\tduration[ms]\n"
        "common_voice_fr_1.mp3\t6048\n"
        "common_voice_fr_2.mp3\t1920\n",
        encoding="utf-8",
    )
    durations = read_clip_durations(tsv)
    assert durations["common_voice_fr_1.mp3"] == 6.048
    assert durations["common_voice_fr_2.mp3"] == 1.920
