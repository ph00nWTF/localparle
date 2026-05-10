"""Durable learner state — JSON-backed, single-user.

Tracks CEFR level, vocabulary seen, error log, and SM-2 spaced-repetition deck
across sessions. Default path: data/learner/me.json.

Citations for field semantics:
  - SM-2 algorithm fields (ReviewCard): Woźniak 1987 — docs/tutor_pedagogy.md §5
  - Feedback type taxonomy (ErrorRecord): Lyster & Ranta 1997 — §3
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

VALID_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ReviewCard:
    word: str
    interval_days: int = 1
    repetition: int = 0
    easiness: float = 2.5
    last_reviewed: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewCard":
        return cls(**d)


@dataclass
class ErrorRecord:
    turn_index: int
    transcript: str
    error_type: str   # gender | agreement | conjugation | false_cognate | other
    word: str
    correction: str
    feedback_type: str  # elicitation | metalinguistic | clarification | recast | explicit | repetition
    timestamp: str

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorRecord":
        return cls(**d)


@dataclass
class LearnerState:
    cefr_level: str
    vocab_seen: dict[str, str] = field(default_factory=dict)        # word -> first-seen ISO
    error_log: list[ErrorRecord] = field(default_factory=list)
    sm2_deck: dict[str, ReviewCard] = field(default_factory=dict)   # word -> card
    scenarios_done: list[str] = field(default_factory=list)
    turn_count: int = 0

    @classmethod
    def default(cls, level: str) -> "LearnerState":
        if level not in VALID_LEVELS:
            raise ValueError(f"level must be one of {VALID_LEVELS}, got {level!r}")
        return cls(cefr_level=level)

    @classmethod
    def from_dict(cls, d: dict) -> "LearnerState":
        return cls(
            cefr_level=d["cefr_level"],
            vocab_seen=dict(d.get("vocab_seen", {})),
            error_log=[ErrorRecord.from_dict(e) for e in d.get("error_log", [])],
            sm2_deck={w: ReviewCard.from_dict(c) for w, c in d.get("sm2_deck", {}).items()},
            scenarios_done=list(d.get("scenarios_done", [])),
            turn_count=int(d.get("turn_count", 0)),
        )


def load(path: str | Path) -> LearnerState | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return LearnerState.from_dict(json.load(f))


def save(state: LearnerState, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(state), f, ensure_ascii=False, indent=2)
