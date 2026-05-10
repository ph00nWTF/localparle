"""Tests for prompt_builder — scaffolding tiers, vocab clamp, SRS injection, JSON schema."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from french_tutor.learner_state import LearnerState, ReviewCard
from french_tutor.prompt_builder import JSON_SCHEMA, build_turn_prompt
from french_tutor.scenarios import pick


@pytest.fixture(scope="session")
def cefr_lookup() -> dict:
    p = Path("data/curriculum/cefr_lookup.json")
    return json.loads(p.read_text(encoding="utf-8"))["words"]


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 4, 29, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Tiered English scaffolding
# ---------------------------------------------------------------------------

def test_a1_prompt_includes_english_gloss_instruction(cefr_lookup, now):
    state = LearnerState.default("A1")
    system, _, _ = build_turn_prompt(
        state, pick("A1"), history=[], transcript="bonjour",
        cefr_lookup=cefr_lookup, now=now,
    )
    assert "traduction anglaise entre parenthèses" in system


def test_a2_prompt_uses_fallback_scaffolding(cefr_lookup, now):
    state = LearnerState.default("A2")
    system, _, _ = build_turn_prompt(
        state, pick("A2"), history=[], transcript="bonjour",
        cefr_lookup=cefr_lookup, now=now,
    )
    assert "uniquement" in system
    assert "deux tours" in system


def test_b1_prompt_forbids_english(cefr_lookup, now):
    state = LearnerState.default("B1")
    system, _, _ = build_turn_prompt(
        state, pick("B1"), history=[], transcript="salut",
        cefr_lookup=cefr_lookup, now=now,
    )
    assert "français uniquement" in system
    assert "Aucun anglais" in system
    assert "parenthèses" not in system  # no scaffolding instruction


# ---------------------------------------------------------------------------
# SRS injection — first turn only
# ---------------------------------------------------------------------------

def test_first_turn_injects_due_review_words(cefr_lookup, now):
    state = LearnerState.default("B1")
    state.sm2_deck["maison"] = ReviewCard(word="maison")  # never reviewed → due
    system, user, _ = build_turn_prompt(
        state, pick("B1"), history=[], transcript="salut",
        cefr_lookup=cefr_lookup, now=now,
    )
    assert "maison" in system
    assert "[début de session]" in user


def test_later_turn_omits_review_words(cefr_lookup, now):
    state = LearnerState.default("B1")
    state.sm2_deck["maison"] = ReviewCard(word="maison")
    state.turn_count = 5
    system, user, _ = build_turn_prompt(
        state, pick("B1"), history=[], transcript="ça va",
        cefr_lookup=cefr_lookup, now=now,
    )
    assert "Au début de cette session" not in system
    assert "[début de session]" not in user


# ---------------------------------------------------------------------------
# JSON schema
# ---------------------------------------------------------------------------

def test_schema_returned_is_the_module_constant(cefr_lookup, now):
    state = LearnerState.default("B1")
    _, _, schema = build_turn_prompt(
        state, pick("B1"), history=[], transcript="salut",
        cefr_lookup=cefr_lookup, now=now,
    )
    assert schema is JSON_SCHEMA


def test_schema_structure():
    assert JSON_SCHEMA["type"] == "object"
    assert set(JSON_SCHEMA["required"]) == {"reply", "errors"}
    assert JSON_SCHEMA["properties"]["reply"]["type"] == "string"
    assert JSON_SCHEMA["properties"]["errors"]["type"] == "array"
    error_item = JSON_SCHEMA["properties"]["errors"]["items"]
    assert set(error_item["required"]) == {"error_type", "word", "correction"}
    assert "gender" in error_item["properties"]["error_type"]["enum"]


# ---------------------------------------------------------------------------
# Scenario context and elicitation rule are present in every prompt
# ---------------------------------------------------------------------------

def test_prompt_includes_elicitation_rule(cefr_lookup, now):
    state = LearnerState.default("B1")
    system, _, _ = build_turn_prompt(
        state, pick("B1"), history=[], transcript="salut",
        cefr_lookup=cefr_lookup, now=now,
    )
    assert "élicitation" in system
    assert "pas de recast" in system
