"""Pure-function pedagogy core. No model, no I/O.

Algorithms cite docs/tutor_pedagogy.md:
  - SM-2 (apply_sm2): Woźniak 1987 — §5
  - Default feedback type (select_feedback_type): Lyster & Ranta 1997 — §3
  - i+1 vocab pool (vocab_pool_for_level): Krashen 1982 — §4
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from .learner_state import (
    VALID_LEVELS,
    ErrorRecord,
    LearnerState,
    ReviewCard,
    now_iso,
)

# Lyster & Ranta 1997 §3: recasts and explicit corrections produced 0% learner-
# generated repair. Elicitation, metalinguistic, clarification, and repetition
# all produced repair. Default = elicitation. Use explicit only after elicitation
# fails twice on the same error word.
ELICITATION_FAILURES_BEFORE_EXPLICIT = 2


def apply_sm2(card: ReviewCard, quality: int, *, now: datetime) -> ReviewCard:
    """Update a review card with a quality grade (0..5).

    Woźniak 1987:
      I(1) := 1, I(2) := 6, I(n) := I(n-1) * EF for n > 2
      EF' := EF + (0.1 - (5-q) * (0.08 + (5-q) * 0.02))
      EF floor = 1.3
      If q < 3: reset (repetition := 0, interval := 1).
    """
    if not 0 <= quality <= 5:
        raise ValueError(f"quality must be 0..5, got {quality}")

    new_easiness = card.easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_easiness = max(1.3, new_easiness)

    if quality < 3:
        new_repetition = 0
        new_interval = 1
    else:
        new_repetition = card.repetition + 1
        if new_repetition == 1:
            new_interval = 1
        elif new_repetition == 2:
            new_interval = 6
        else:
            new_interval = max(1, round(card.interval_days * new_easiness))

    return replace(
        card,
        repetition=new_repetition,
        interval_days=new_interval,
        easiness=new_easiness,
        last_reviewed=now.isoformat(),
    )


def next_due_cards(state: LearnerState, *, now: datetime, k: int = 5) -> list[ReviewCard]:
    """Return up to k cards due for review, oldest-due first.

    A card is due when last_reviewed + interval_days <= now. Cards never reviewed
    (last_reviewed == "") are always due.
    """
    due: list[tuple[datetime, ReviewCard]] = []
    for card in state.sm2_deck.values():
        if not card.last_reviewed:
            due.append((datetime.min.replace(tzinfo=now.tzinfo), card))
            continue
        last = datetime.fromisoformat(card.last_reviewed)
        next_due = last + timedelta(days=card.interval_days)
        if next_due <= now:
            due.append((next_due, card))
    due.sort(key=lambda t: t[0])
    return [c for _, c in due[:k]]


def select_feedback_type(error_type: str, prior_failures: int) -> str:
    """Pick a corrective-feedback strategy.

    Default: elicitation. After ELICITATION_FAILURES_BEFORE_EXPLICIT misses on
    the same error, fall back to explicit. Lyster & Ranta found recasts produce
    no repair, so we never recommend recast.
    """
    if prior_failures >= ELICITATION_FAILURES_BEFORE_EXPLICIT:
        return "explicit"
    if error_type == "false_cognate":
        # Specific lexical knowledge gap — metalinguistic clue more useful than elicitation alone.
        return "metalinguistic"
    return "elicitation"


def update_after_turn(
    state: LearnerState,
    transcript: str,
    errors: list[dict],
    *,
    now: datetime,
) -> LearnerState:
    """Mutate state after a learner turn. Returns the same state object.

    Inputs:
      transcript: what the learner said (already ASR-decoded).
      errors: list of {error_type, word, correction} dicts emitted by the LLM.

    Side effects:
      - Records first-seen timestamp for each unique word in transcript.
      - Appends an ErrorRecord for each error.
      - Adds a fresh ReviewCard to sm2_deck for each error word not already there.
      - Increments turn_count.
    """
    state.turn_count += 1
    iso = now.isoformat()

    for word in _tokenize(transcript):
        state.vocab_seen.setdefault(word, iso)

    for err in errors:
        word = err["word"].lower()
        prior_failures = sum(
            1 for e in state.error_log if e.word == word and e.feedback_type == "elicitation"
        )
        feedback_type = select_feedback_type(err["error_type"], prior_failures)
        state.error_log.append(ErrorRecord(
            turn_index=state.turn_count,
            transcript=transcript,
            error_type=err["error_type"],
            word=word,
            correction=err["correction"],
            feedback_type=feedback_type,
            timestamp=iso,
        ))
        if word not in state.sm2_deck:
            state.sm2_deck[word] = ReviewCard(word=word)

    return state


def vocab_pool_for_level(
    cefr_lookup: dict,
    level: str,
    *,
    stretch_words: int = 50,
) -> list[str]:
    """Return the vocab pool a tutor reply may freely draw from.

    Krashen i+1: include all words at or below `level`, plus a stretch sample
    from the next level up. The stretch is bounded so prompts stay short.
    """
    if level not in VALID_LEVELS:
        raise ValueError(f"level must be one of {VALID_LEVELS}, got {level!r}")
    cutoff = VALID_LEVELS.index(level)
    next_level = VALID_LEVELS[cutoff + 1] if cutoff + 1 < len(VALID_LEVELS) else None

    in_pool: list[str] = []
    stretch: list[str] = []
    for word, entry in cefr_lookup.items():
        word_level = entry.get("level")
        if word_level is None:
            continue
        if word_level in VALID_LEVELS[: cutoff + 1]:
            in_pool.append(word)
        elif next_level and word_level == next_level and len(stretch) < stretch_words:
            stretch.append(word)
    return in_pool + stretch


def english_scaffold_for_level(level: str) -> str:
    """Map CEFR level → scaffolding policy used in prompt construction.

    Returns one of: "free" | "fallback" | "none".
      free     — A1: gloss new vocabulary in parentheses freely.
      fallback — A2: gloss only when negotiation has failed.
      none     — B1+: French only.
    """
    if level == "A1":
        return "free"
    if level == "A2":
        return "fallback"
    return "none"


# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    out: list[str] = []
    word = []
    for ch in text.lower():
        if ch.isalpha() or ch in "'-":
            word.append(ch)
        elif word:
            out.append("".join(word))
            word = []
    if word:
        out.append("".join(word))
    return [w for w in out if len(w) >= 2]


# Re-export so callers can `from french_tutor.pedagogy import now_iso`.
__all__ = [
    "apply_sm2",
    "english_scaffold_for_level",
    "next_due_cards",
    "now_iso",
    "select_feedback_type",
    "update_after_turn",
    "vocab_pool_for_level",
]
