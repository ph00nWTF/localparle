"""End-to-end glue test for ConversationPipeline with all engines mocked."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from french_tutor.learner_state import LearnerState, load
from french_tutor.scenarios import pick
from french_tutor.tutor import ConversationPipeline


class FakeASR:
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[str, float]:
        return "le maison est grand", 1.0


class FakeLLM:
    def __init__(self):
        self.calls: list[dict] = []

    def respond(self, system: str, user: str, history: list, schema: dict) -> dict:
        self.calls.append({"system": system, "user": user, "history": list(history)})
        return {
            "reply": "Salut ! Le ou la maison ?",
            "errors": [{"error_type": "gender", "word": "maison", "correction": "la maison"}],
        }


class FakeTTS:
    def synthesize(self, text: str) -> bytes:
        return b"RIFF....WAVE" + text.encode("utf-8")


@pytest.fixture
def cefr_lookup() -> dict:
    return json.loads(Path("data/curriculum/cefr_lookup.json").read_text(encoding="utf-8"))["words"]


def _make_pipeline(state_path: Path, cefr_lookup: dict) -> ConversationPipeline:
    return ConversationPipeline(
        asr=FakeASR(),
        llm=FakeLLM(),
        tts=FakeTTS(),
        state=LearnerState.default("B1"),
        scenario=pick("B1"),
        cefr_lookup=cefr_lookup,
        state_path=state_path,
    )


def test_pipeline_turn_orchestration(tmp_path: Path, cefr_lookup):
    pipeline = _make_pipeline(tmp_path / "me.json", cefr_lookup)
    audio = np.zeros(16000, dtype=np.float32)
    result = pipeline.turn(audio)

    assert result.transcript == "le maison est grand"
    assert "maison" in result.response_text
    assert result.response_wav.startswith(b"RIFF")
    assert result.errors[0]["word"] == "maison"
    assert len(pipeline.history) == 2  # user + assistant


def test_pipeline_persists_state(tmp_path: Path, cefr_lookup):
    state_path = tmp_path / "me.json"
    pipeline = _make_pipeline(state_path, cefr_lookup)
    pipeline.turn(np.zeros(16000, dtype=np.float32))

    saved = load(state_path)
    assert saved is not None
    assert saved.turn_count == 1
    assert "maison" in saved.sm2_deck
    assert any(e.error_type == "gender" for e in saved.error_log)


def test_pipeline_reset_clears_history(tmp_path: Path, cefr_lookup):
    pipeline = _make_pipeline(tmp_path / "me.json", cefr_lookup)
    pipeline.turn(np.zeros(16000, dtype=np.float32))
    pipeline.reset()
    assert pipeline.history == []

