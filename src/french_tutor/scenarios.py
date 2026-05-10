"""Conversation scenarios — anchors for immersion practice.

Scenarios live as JSON in data/curriculum/scenarios.json. This module is just
the loader.
"""
from __future__ import annotations

import json
from pathlib import Path

_SCENARIOS_PATH = Path(__file__).parents[2] / "data" / "curriculum" / "scenarios.json"


def load_scenarios() -> list[dict]:
    return json.loads(_SCENARIOS_PATH.read_text(encoding="utf-8"))["scenarios"]


def pick(level: str) -> dict:
    """Return the first scenario at the given CEFR level. Raises if none match."""
    for s in load_scenarios():
        if s["level"] == level:
            return s
    raise LookupError(f"no scenario for level {level!r}")
