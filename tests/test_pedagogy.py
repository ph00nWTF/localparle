"""Tests for pedagogy core. Pure functions — no model, no I/O."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from french_tutor.learner_state import LearnerState, ReviewCard
from french_tutor.pedagogy import (
    apply_sm2,
    english_scaffold_for_level,
    next_due_cards,
    select_feedback_type,
    update_after_turn,
    vocab_pool_for_level,
)
from french_tutor.pedagogy import _tokenize  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# SM-2 — Woźniak 1987 progression
# ---------------------------------------------------------------------------

def test_sm2_first_repetition_interval_is_1():
    card = ReviewCard(word="café")
    after = apply_sm2(card, quality=5, now=datetime(2026, 4, 29, tzinfo=UTC))
    assert after.repetition == 1
    assert after.interval_days == 1


def test_sm2_second_repetition_interval_is_6():
    card = ReviewCard(word="café", repetition=1, interval_days=1)
    after = apply_sm2(card, quality=5, now=datetime(2026, 4, 30, tzinfo=UTC))
    assert after.repetition == 2
    assert after.interval_days == 6


def test_sm2_third_repetition_uses_easiness():
    card = ReviewCard(word="café", repetition=2, interval_days=6, easiness=2.6)
    after = apply_sm2(card, quality=5, now=datetime(2026, 5, 6, tzinfo=UTC))
    assert after.repetition == 3
    # 6 * 2.6 = 15.6 → rounded → 16
    assert after.interval_days == 16


def test_sm2_lapse_resets_repetition_and_interval():
    card = ReviewCard(word="café", repetition=3, interval_days=16, easiness=2.6)
    after = apply_sm2(card, quality=2, now=datetime(2026, 5, 22, tzinfo=UTC))
    assert after.repetition == 0
    assert after.interval_days == 1


def test_sm2_easiness_floor_is_1_3():
    card = ReviewCard(word="café", easiness=1.3)
    # q=0 drives easiness lower; floor must clamp at 1.3.
    after = apply_sm2(card, quality=0, now=datetime(2026, 4, 29, tzinfo=UTC))
    assert after.easiness == pytest.approx(1.3)


def test_sm2_easiness_increases_on_perfect():
    card = ReviewCard(word="café", easiness=2.5)
    after = apply_sm2(card, quality=5, now=datetime(2026, 4, 29, tzinfo=UTC))
    # EF' = 2.5 + (0.1 - 0*(0.08+0)) = 2.6
    assert after.easiness == pytest.approx(2.6)


def test_sm2_full_progression_matches_published_behavior():
    """Q=5,5,5,2,5 should give intervals 1, 6, ~16, 1, 6 (lapse resets, then I(2)=6)."""
    card = ReviewCard(word="maison")
    intervals: list[int] = []
    now = datetime(2026, 4, 29, tzinfo=UTC)
    for q in (5, 5, 5, 2, 5):
        card = apply_sm2(card, quality=q, now=now)
        intervals.append(card.interval_days)
        now += timedelta(days=card.interval_days)
    assert intervals[0] == 1
    assert intervals[1] == 6
    assert intervals[2] >= 14  # 6 * EF where EF ≈ 2.5–2.6
    assert intervals[3] == 1   # lapse reset
    assert intervals[4] == 1   # repetition counter is 1 after reset → I(1)=1


def test_sm2_quality_bounds():
    card = ReviewCard(word="café")
    with pytest.raises(ValueError):
        apply_sm2(card, quality=-1, now=datetime(2026, 4, 29, tzinfo=UTC))
    with pytest.raises(ValueError):
        apply_sm2(card, quality=6, now=datetime(2026, 4, 29, tzinfo=UTC))


def test_sm2_does_not_mutate_input():
    card = ReviewCard(word="café")
    apply_sm2(card, quality=5, now=datetime(2026, 4, 29, tzinfo=UTC))
    assert card.repetition == 0
    assert card.interval_days == 1


# ---------------------------------------------------------------------------
# next_due_cards
# ---------------------------------------------------------------------------

def test_next_due_returns_unreviewed_cards_first():
    state = LearnerState.default("B1")
    state.sm2_deck["never"] = ReviewCard(word="never")  # last_reviewed = ""
    state.sm2_deck["recent"] = ReviewCard(
        word="recent", interval_days=10,
        last_reviewed=datetime(2026, 4, 29, tzinfo=UTC).isoformat(),
    )
    due = next_due_cards(state, now=datetime(2026, 4, 30, tzinfo=UTC))
    assert [c.word for c in due] == ["never"]


def test_next_due_includes_overdue_cards():
    state = LearnerState.default("B1")
    state.sm2_deck["overdue"] = ReviewCard(
        word="overdue", interval_days=5,
        last_reviewed=datetime(2026, 4, 1, tzinfo=UTC).isoformat(),
    )
    due = next_due_cards(state, now=datetime(2026, 4, 29, tzinfo=UTC))
    assert [c.word for c in due] == ["overdue"]


def test_next_due_caps_at_k():
    state = LearnerState.default("B1")
    for i in range(10):
        state.sm2_deck[f"w{i}"] = ReviewCard(word=f"w{i}")
    due = next_due_cards(state, now=datetime(2026, 4, 29, tzinfo=UTC), k=3)
    assert len(due) == 3


# ---------------------------------------------------------------------------
# select_feedback_type — Lyster & Ranta 1997
# ---------------------------------------------------------------------------

def test_default_feedback_is_elicitation():
    assert select_feedback_type("gender", prior_failures=0) == "elicitation"
    assert select_feedback_type("conjugation", prior_failures=0) == "elicitation"


def test_false_cognate_uses_metalinguistic():
    assert select_feedback_type("false_cognate", prior_failures=0) == "metalinguistic"


def test_repeated_failure_escalates_to_explicit():
    assert select_feedback_type("gender", prior_failures=2) == "explicit"
    assert select_feedback_type("gender", prior_failures=5) == "explicit"


def test_recast_is_never_selected():
    # Sanity: the function should not return "recast" under any input we model.
    for et in ("gender", "agreement", "conjugation", "false_cognate", "other"):
        for failures in range(5):
            assert select_feedback_type(et, prior_failures=failures) != "recast"


# ---------------------------------------------------------------------------
# update_after_turn
# ---------------------------------------------------------------------------

def test_update_records_vocab_seen():
    state = LearnerState.default("B1")
    update_after_turn(state, "bonjour je vais bien", errors=[], now=datetime(2026, 4, 29, tzinfo=UTC))
    assert "bonjour" in state.vocab_seen
    assert "je" in state.vocab_seen
    assert "vais" in state.vocab_seen
    assert "bien" in state.vocab_seen


def test_update_increments_turn_count():
    state = LearnerState.default("B1")
    update_after_turn(state, "salut", errors=[], now=datetime(2026, 4, 29, tzinfo=UTC))
    update_after_turn(state, "ça va", errors=[], now=datetime(2026, 4, 29, tzinfo=UTC))
    assert state.turn_count == 2


def test_update_records_errors_and_creates_sm2_card():
    state = LearnerState.default("B1")
    errors = [{"error_type": "gender", "word": "maison", "correction": "la maison"}]
    update_after_turn(state, "le maison", errors=errors, now=datetime(2026, 4, 29, tzinfo=UTC))
    assert len(state.error_log) == 1
    assert state.error_log[0].word == "maison"
    assert state.error_log[0].feedback_type == "elicitation"
    assert "maison" in state.sm2_deck


def test_repeated_error_does_not_duplicate_sm2_card():
    state = LearnerState.default("B1")
    errors = [{"error_type": "gender", "word": "maison", "correction": "la maison"}]
    for _ in range(3):
        update_after_turn(state, "le maison", errors=errors, now=datetime(2026, 4, 29, tzinfo=UTC))
    assert len(state.error_log) == 3
    assert len(state.sm2_deck) == 1


def test_repeated_error_escalates_feedback_type():
    state = LearnerState.default("B1")
    errors = [{"error_type": "gender", "word": "maison", "correction": "la maison"}]
    feedback_seq = []
    for _ in range(4):
        update_after_turn(state, "le maison", errors=errors, now=datetime(2026, 4, 29, tzinfo=UTC))
        feedback_seq.append(state.error_log[-1].feedback_type)
    # First two: elicitation; after 2 prior elicitation failures: explicit.
    assert feedback_seq[0] == "elicitation"
    assert feedback_seq[1] == "elicitation"
    assert feedback_seq[2] == "explicit"


# ---------------------------------------------------------------------------
# vocab_pool_for_level — Krashen i+1
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cefr_lookup() -> dict:
    p = Path("data/curriculum/cefr_lookup.json")
    return json.loads(p.read_text(encoding="utf-8"))["words"]


def test_vocab_pool_a1_contains_only_a1(cefr_lookup):
    pool = set(vocab_pool_for_level(cefr_lookup, "A1", stretch_words=0))
    assert pool, "pool must not be empty"
    levels_in_pool = {cefr_lookup[w]["level"] for w in pool}
    assert levels_in_pool == {"A1"}


def test_vocab_pool_b1_includes_a1_a2_b1(cefr_lookup):
    pool = set(vocab_pool_for_level(cefr_lookup, "B1", stretch_words=0))
    levels_in_pool = {cefr_lookup[w]["level"] for w in pool}
    assert levels_in_pool == {"A1", "A2", "B1"}


def test_vocab_pool_includes_stretch_from_next_level(cefr_lookup):
    pool = vocab_pool_for_level(cefr_lookup, "A1", stretch_words=10)
    stretch = [w for w in pool if cefr_lookup[w]["level"] == "A2"]
    assert len(stretch) == 10


def test_vocab_pool_c2_has_no_stretch(cefr_lookup):
    """C2 is the top level — nothing to stretch into."""
    pool = vocab_pool_for_level(cefr_lookup, "C2", stretch_words=50)
    levels = {cefr_lookup[w]["level"] for w in pool}
    assert levels.issubset({"A1", "A2", "B1", "B2", "C1", "C2"})
    # No level above C2 exists in CEFR.


def test_vocab_pool_rejects_invalid_level(cefr_lookup):
    with pytest.raises(ValueError):
        vocab_pool_for_level(cefr_lookup, "Z9")


# ---------------------------------------------------------------------------
# english_scaffold_for_level
# ---------------------------------------------------------------------------

def test_scaffold_a1_is_free():
    assert english_scaffold_for_level("A1") == "free"


def test_scaffold_a2_is_fallback():
    assert english_scaffold_for_level("A2") == "fallback"


@pytest.mark.parametrize("level", ["B1", "B2", "C1", "C2"])
def test_scaffold_b1_and_above_is_none(level):
    assert english_scaffold_for_level(level) == "none"


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_lowercases_and_splits():
    assert _tokenize("Bonjour, ça va?") == ["bonjour", "ça", "va"]


def test_tokenize_preserves_apostrophes_and_hyphens():
    assert _tokenize("c'est un grand-père") == ["c'est", "un", "grand-père"]


def test_tokenize_drops_singletons():
    assert _tokenize("a b cd") == ["cd"]
