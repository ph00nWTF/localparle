"""Tests for learner_state — JSON round-trip, defaults, validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from french_tutor.learner_state import (
    VALID_LEVELS,
    ErrorRecord,
    LearnerState,
    ReviewCard,
    load,
    now_iso,
    save,
)


def test_default_state_has_only_level():
    s = LearnerState.default("B1")
    assert s.cefr_level == "B1"
    assert s.vocab_seen == {}
    assert s.error_log == []
    assert s.sm2_deck == {}
    assert s.scenarios_done == []
    assert s.turn_count == 0


def test_default_rejects_invalid_level():
    with pytest.raises(ValueError):
        LearnerState.default("Z9")


@pytest.mark.parametrize("level", VALID_LEVELS)
def test_default_accepts_all_cefr_levels(level: str):
    assert LearnerState.default(level).cefr_level == level


def test_round_trip_empty(tmp_path: Path):
    s = LearnerState.default("A2")
    p = tmp_path / "me.json"
    save(s, p)
    assert load(p) == s


def test_round_trip_populated(tmp_path: Path):
    s = LearnerState.default("B1")
    s.vocab_seen["bonjour"] = now_iso()
    s.sm2_deck["maison"] = ReviewCard(word="maison", interval_days=6, repetition=2, easiness=2.6, last_reviewed=now_iso())
    s.error_log.append(ErrorRecord(
        turn_index=1, transcript="le maison",
        error_type="gender", word="maison", correction="la maison",
        feedback_type="elicitation", timestamp=now_iso(),
    ))
    s.scenarios_done.append("cafe_a1")
    s.turn_count = 1
    p = tmp_path / "me.json"
    save(s, p)
    assert load(p) == s


def test_load_missing_returns_none(tmp_path: Path):
    assert load(tmp_path / "nope.json") is None


def test_save_creates_parent_dirs(tmp_path: Path):
    p = tmp_path / "deep" / "nested" / "me.json"
    save(LearnerState.default("A1"), p)
    assert p.exists()


def test_save_writes_utf8_unicode(tmp_path: Path):
    s = LearnerState.default("B1")
    s.vocab_seen["café"] = now_iso()
    p = tmp_path / "me.json"
    save(s, p)
    raw = p.read_text(encoding="utf-8")
    assert "café" in raw  # ensure_ascii=False preserves accents


def test_json_shape_is_stable(tmp_path: Path):
    s = LearnerState.default("B2")
    p = tmp_path / "me.json"
    save(s, p)
    parsed = json.loads(p.read_text(encoding="utf-8"))
    assert set(parsed.keys()) == {
        "cefr_level", "vocab_seen", "error_log",
        "sm2_deck", "scenarios_done", "turn_count",
    }
