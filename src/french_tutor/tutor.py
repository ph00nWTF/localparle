"""ASR + LLM + TTS engines and the voice-mode ConversationPipeline.

Text mode (scripts/tutor_text.py) orchestrates the LLM directly without going
through ConversationPipeline — there's no ASR or TTS to wrap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .learner_state import LearnerState, save
from .pedagogy import update_after_turn
from .prompt_builder import build_turn_prompt


@dataclass
class TurnResult:
    transcript: str
    response_text: str
    response_wav: bytes
    errors: list[dict] = field(default_factory=list)


_BASELINE_MODEL_ID = "facebook/wav2vec2-large-xlsr-53-french"
_ASR_ALIASES = {"baseline": _BASELINE_MODEL_ID}


class ASREngine:
    """wav2vec2 transcription. `checkpoint_dir` is one of:
      - a local path (e.g. "models/final")  → loaded from disk
      - "baseline"                           → facebook/wav2vec2-large-xlsr-53-french
      - any HF model ID (contains "/")      → pulled by from_pretrained
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "baseline",
        device: str | None = None,
        lm_path: str | Path | None = None,
        unigrams_path: str | Path | None = None,
    ):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        source = _ASR_ALIASES.get(str(checkpoint_dir), str(checkpoint_dir))
        # Local path with no weights → fall back to baseline rather than crash.
        p = Path(source)
        if p.exists() and not (any(p.glob("*.safetensors")) or any(p.glob("*.bin"))):
            source = _BASELINE_MODEL_ID

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(source)
        self.model = Wav2Vec2ForCTC.from_pretrained(source, use_safetensors=True).to(self.device).eval()
        self.source = source

        self.decoder = None
        if lm_path:
            from pyctcdecode import build_ctcdecoder

            vocab_dict = self.processor.tokenizer.get_vocab()
            # kenlm/unigrams are trained on lowercase French text; the wav2vec2
            # vocab is uppercase. Lowercase labels so the per-frame characters
            # align with what the LM scores. (Standard HF/pyctcdecode pattern.)
            labels = [t.lower() for t, _ in sorted(vocab_dict.items(), key=lambda kv: kv[1])]
            unigrams = None
            if unigrams_path:
                unigrams = Path(unigrams_path).read_text(encoding="utf-8").splitlines()
            self.decoder = build_ctcdecoder(labels, str(lm_path), unigrams=unigrams)
            self.lm_path = str(lm_path)

    _MAX_CHUNK_S = 15  # wav2vec2 attention degrades past ~15 s; split longer audio

    def _split_chunks(self, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        """Split audio into ≤_MAX_CHUNK_S silence-bounded chunks."""
        max_samples = self._MAX_CHUNK_S * sample_rate
        if len(audio) <= max_samples:
            return [audio]

        import webrtcvad
        frame_size = int(sample_rate * 0.03)  # 30 ms frames
        pcm = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
        vad = webrtcvad.Vad(2)
        n_frames = len(pcm) // frame_size
        is_speech = [
            vad.is_speech(pcm[i * frame_size:(i + 1) * frame_size].tobytes(), sample_rate)
            for i in range(n_frames)
        ]

        chunks, start = [], 0
        for i in range(n_frames):
            end_sample = (i + 1) * frame_size
            if end_sample - start >= max_samples and not is_speech[i]:
                chunks.append(audio[start:end_sample])
                start = end_sample
        if start < len(audio):
            chunks.append(audio[start:])
        return chunks if chunks else [audio]

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[str, float]:
        """Returns (transcript, confidence in [0,1]).

        Confidence is the mean of per-frame max softmax over non-blank frames —
        a standard CTC confidence proxy. Low values flag "I didn't catch that."
        Long audio (>15 s) is split on silence boundaries and decoded chunk-by-chunk.
        """
        import torch
        import torch.nn.functional as F

        from .data import normalize_transcript

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        chunks = self._split_chunks(audio, sample_rate)
        texts, confidences = [], []
        for chunk in chunks:
            inputs = self.processor(chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self.model(
                    inputs.input_values.to(self.device),
                    attention_mask=inputs.attention_mask.to(self.device),
                ).logits

            probs = F.softmax(logits, dim=-1)[0]
            max_probs, ids_per_frame = probs.max(dim=-1)
            blank_id = self.processor.tokenizer.pad_token_id
            non_blank = ids_per_frame != blank_id
            confidences.append(float(max_probs[non_blank].mean()) if non_blank.any() else 0.0)

            if self.decoder is not None:
                texts.append(self.decoder.decode(logits.cpu().numpy()[0]))
            else:
                ids = torch.argmax(logits, dim=-1)
                texts.append(self.processor.batch_decode(ids)[0])

        transcript = normalize_transcript(" ".join(texts))
        confidence = float(np.mean(confidences)) if confidences else 0.0
        return transcript, confidence


class EnglishASREngine:
    """faster-whisper tiny (multilingual) — language detection + English transcription."""

    def __init__(self, model_size: str = "tiny", device: str = "cuda"):
        from faster_whisper import WhisperModel
        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute)

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        segments, info = self.model.transcribe(audio)
        if info.language != "en":
            return "", 0.0
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, float(info.language_probability)


class LLMEngine:
    """Talks to local Ollama via /api/chat with JSON-schema-constrained output."""

    def __init__(
        self,
        *,
        ollama_url: str,
        model: str,
        temperature: float = 0.7,
        timeout_s: float = 60.0,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s

    def respond(
        self,
        system: str,
        user: str,
        history: list[dict],
        schema: dict,
    ) -> dict:
        """Call Ollama /api/chat with format=schema. Returns parsed JSON dict.

        history items are {"role": "user"|"assistant", "content": str}.
        Returns the schema-conformant response: {"reply": str, "errors": [...]}.
        """
        import httpx

        messages: list[dict] = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {"temperature": self.temperature},
        }
        r = httpx.post(f"{self.ollama_url}/api/chat", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return json.loads(content)

    def generate_opener(self, level: str) -> dict:
        """Ask the model for a random conversational opener at this CEFR level.

        Returns {topic, setting, opening_line}. Used at session start to avoid
        hard-coded scenarios. Higher temperature for variety.
        """
        import httpx

        schema = {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "setting": {"type": "string"},
                "opening_line": {"type": "string"},
            },
            "required": ["topic", "setting", "opening_line"],
        }
        sys = (
            f"Tu démarres une conversation en français avec un apprenant de niveau {level}. "
            "Choisis un SUJET ALÉATOIRE inattendu — pas toujours un café ou un restaurant. "
            "Idées variées : la météo, un voyage récent, un film, un animal de compagnie, "
            "le sport, un livre, la cuisine, la musique, un souvenir d'enfance, le travail, "
            "un hobby, un problème technique, un voisin, une fête. "
            "Renvoie un JSON {topic, setting (1 phrase qui pose le contexte), "
            "opening_line (1 phrase pour démarrer, en français, niveau approprié)}."
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys},
                         {"role": "user", "content": "Démarre."}],
            "stream": False,
            "format": schema,
            "options": {"temperature": 1.0},
        }
        r = httpx.post(f"{self.ollama_url}/api/chat", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])


class TTSEngine:
    """Piper neural TTS via the piper-tts Python package. Returns WAV bytes."""

    def __init__(self, *, voice_path: str | Path):
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(voice_path))

    def synthesize(self, text: str) -> bytes:
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            self.voice.synthesize_wav(text, w)
        return buf.getvalue()



@dataclass
class ConversationPipeline:
    """Voice-mode orchestrator. ASR → prompt → LLM → state update → TTS."""

    asr: Any           # ASREngine or fake
    llm: Any           # LLMEngine or fake
    tts: Any           # TTSEngine or fake
    state: LearnerState
    scenario: dict
    cefr_lookup: dict
    state_path: Path
    history: list[dict] = field(default_factory=list)

    def turn(self, audio: np.ndarray, sample_rate: int = 16000) -> TurnResult:
        transcript, _confidence = self.asr.transcribe(audio, sample_rate=sample_rate)
        now = datetime.now(UTC)
        system, user, schema = build_turn_prompt(
            self.state, self.scenario, self.history, transcript,
            cefr_lookup=self.cefr_lookup, now=now,
        )
        result = self.llm.respond(system, user, self.history, schema)
        update_after_turn(self.state, transcript, result.get("errors", []), now=now)
        save(self.state, self.state_path)
        wav = self.tts.synthesize(result["reply"])
        self.history.append({"role": "user", "content": transcript})
        self.history.append({"role": "assistant", "content": result["reply"]})
        return TurnResult(
            transcript=transcript,
            response_text=result["reply"],
            response_wav=wav,
            errors=result.get("errors", []),
        )

    def reset(self) -> None:
        self.history.clear()



def build_pipeline(tutor_cfg: dict) -> ConversationPipeline:
    """Construct ASR/LLM/TTS engines and assemble the pipeline from configs/tutor.yaml dict."""
    from .learner_state import LearnerState
    from .learner_state import load as load_state
    from .prompt_builder import load_cefr_lookup

    asr_cfg = tutor_cfg["asr"]
    llm_cfg = tutor_cfg["llm"]
    tts_cfg = tutor_cfg["tts"]
    ped_cfg = tutor_cfg.get("pedagogy", {})

    asr = ASREngine(
        checkpoint_dir=asr_cfg["checkpoint_dir"],
        device=asr_cfg.get("device"),
        lm_path=asr_cfg.get("lm_path"),
        unigrams_path=asr_cfg.get("unigrams_path"),
    )
    llm = LLMEngine(
        ollama_url=llm_cfg["ollama_url"],
        model=llm_cfg["model"],
        temperature=llm_cfg.get("temperature", 0.7),
    )
    tts = TTSEngine(voice_path=tts_cfg["voice_path"])

    state_path = Path(ped_cfg.get("learner_state_path", "data/learner/me.json"))
    state = load_state(state_path) or LearnerState.default(ped_cfg.get("default_level", "B1"))
    cefr_lookup = load_cefr_lookup()
    scenario = llm.generate_opener(state.cefr_level)

    return ConversationPipeline(
        asr=asr, llm=llm, tts=tts,
        state=state, scenario=scenario,
        cefr_lookup=cefr_lookup, state_path=state_path,
    )
