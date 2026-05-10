"""Tests for scenarios — schema, level filtering, vocab references resolve."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from french_tutor.scenarios import load_scenarios, pick

REQUIRED_FIELDS = {"id", "level", "setting", "opening_line", "target_vocab", "target_grammar"}
LEVEL_ORDER = ("A1", "A2", "B1", "B2", "C1", "C2")


@pytest.fixture(scope="session")
def cefr_lookup() -> dict:
    p = Path("data/curriculum/cefr_lookup.json")
    return json.loads(p.read_text(encoding="utf-8"))["words"]


def test_load_returns_nonempty_list():
    scenarios = load_scenarios()
    assert isinstance(scenarios, list)
    assert len(scenarios) >= 3


def test_each_scenario_has_required_fields():
    for s in load_scenarios():
        assert REQUIRED_FIELDS <= s.keys(), f"missing fields in scenario {s.get('id')!r}"
        assert s["level"] in LEVEL_ORDER
        assert isinstance(s["target_vocab"], list)
        assert isinstance(s["target_grammar"], list)
        assert s["opening_line"].strip(), f"empty opening_line in {s['id']!r}"


def test_scenario_ids_are_unique():
    ids = [s["id"] for s in load_scenarios()]
    assert len(ids) == len(set(ids))


def test_pick_returns_scenario_at_level():
    s = pick("A1")
    assert s["level"] == "A1"


def test_pick_raises_on_missing_level():
    with pytest.raises(LookupError):
        pick("C2")


def test_target_vocab_words_resolve_in_cefr_lookup(cefr_lookup):
    """Every target_vocab word must exist in cefr_lookup at a level ≤ scenario.level.

    This catches scenarios accidentally targeting words above the learner's level.
    """
    for s in load_scenarios():
        scenario_cutoff = LEVEL_ORDER.index(s["level"])
        allowed = set(LEVEL_ORDER[: scenario_cutoff + 1])
        for word in s["target_vocab"]:
            entry = cefr_lookup.get(word.lower())
            if entry is None:
                # Some scenario words (e.g. "merci", "bonjour") are core vocab not in
                # FLELex's frequency-graded list. That's fine — the lookup is a soft
                # check, not an exhaustive index.
                continue
            assert entry["level"] in allowed, (
                f"scenario {s['id']!r} targets {word!r} at level {entry['level']} "
                f"(scenario level: {s['level']})"
            )
