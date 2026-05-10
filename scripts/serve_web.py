"""Tiny web server for the French tutor — old-school stdlib http.server.

Returns text + base64 WAV per turn so the browser can play the tutor's reply.

    uv run python scripts/serve_web.py --config configs/tutor.yaml

Routes:
  GET  /              static/index.html
  GET  /static/<path> static asset (CSS, JS if any)
  POST /turn          body = audio bytes (Content-Type: audio/webm) OR
                             JSON {"text": "..."}; returns JSON {transcript, reply, errors}
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import subprocess
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from french_tutor.config import load_yaml
from french_tutor.learner_state import VALID_LEVELS, LearnerState, load, save
from french_tutor.pedagogy import update_after_turn
from french_tutor.prompt_builder import build_turn_prompt, load_cefr_lookup
from french_tutor.tutor import ASREngine, EnglishASREngine, LLMEngine, TTSEngine
from french_tutor.utils import setup_logging

log = logging.getLogger("serve_web")

ROOT = Path(__file__).parents[1]
STATIC_DIR = ROOT / "static"


class App:
    """Single-process state. http handlers reach in to call .turn()."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state_path = Path(cfg["pedagogy"]["learner_state_path"])
        self.state = load(self.state_path) or LearnerState.default(cfg["pedagogy"]["default_level"])
        save(self.state, self.state_path)
        self.cefr_lookup = load_cefr_lookup()

        log.info("loading ASR…")
        self.asr = ASREngine(
            checkpoint_dir=cfg["asr"]["checkpoint_dir"],
            lm_path=cfg["asr"].get("lm_path"),
            unigrams_path=cfg["asr"].get("unigrams_path"),
        )
        log.info("ASR ready (source=%s, lm=%s)", self.asr.source, getattr(self.asr, "lm_path", None))

        self.llm = LLMEngine(
            ollama_url=cfg["llm"]["ollama_url"],
            model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
        )
        log.info("LLM endpoint configured for model=%s", self.llm.model)

        log.info("loading TTS…")
        self.tts = TTSEngine(voice_path=cfg["tts"]["voice_path"])
        log.info("TTS ready (%s)", cfg["tts"]["voice_path"])

        self.confidence_threshold = float(cfg["asr"].get("confidence_threshold", 0.6))
        self.dual_asr_threshold = float(cfg["asr"].get("dual_asr_threshold", 0.75))

        log.info("loading English ASR…")
        self.en_asr = EnglishASREngine(device="cuda" if cfg["asr"].get("device", "cuda") == "cuda" else "cpu")
        log.info("English ASR ready")

        self.mode = "guided"

        log.info("generating opener…")
        self.scenario = self.llm.generate_opener(self.state.cefr_level)
        log.info("opener: %s — %s", self.scenario["topic"], self.scenario["opening_line"])
        self.history = [{"role": "assistant", "content": self.scenario["opening_line"]}]
        self.opening_wav = self.tts.synthesize(self.scenario["opening_line"])

    def turn(self, transcript: str, confidence: float = 1.0,
             en_context: str | None = None, en_dominant: bool = False) -> dict:
        # Low-confidence ASR or near-empty transcript → clarification, no LLM call.
        # Standard pattern in production language tutors (Speak.com, dialogue-system
        # research): never let the model confidently respond to garbage.
        if confidence < self.confidence_threshold or len(transcript.strip()) < 3:
            clarif = "Je n'ai pas bien compris, tu peux répéter ?"
            wav = self.tts.synthesize(clarif)
            log.info("low confidence (%.2f) — asking clarification", confidence)
            return {
                "transcript": transcript,
                "reply": clarif,
                "errors": [],
                "low_confidence": True,
                "reply_audio_b64": base64.b64encode(wav).decode("ascii"),
            }

        now = datetime.now(UTC)
        system, user, schema = build_turn_prompt(
            self.state, self.scenario, self.history, transcript,
            cefr_lookup=self.cefr_lookup, now=now, mode=self.mode,
            en_context=en_context, en_dominant=en_dominant,
        )
        result = self.llm.respond(system, user, self.history, schema)
        update_after_turn(self.state, transcript, result.get("errors", []), now=now)
        save(self.state, self.state_path)
        self.history.append({"role": "user", "content": transcript})
        self.history.append({"role": "assistant", "content": result["reply"]})
        wav = self.tts.synthesize(result["reply"])
        return {
            "transcript": transcript,
            "reply": result["reply"],
            "errors": result.get("errors", []),
            "reply_audio_b64": base64.b64encode(wav).decode("ascii"),
        }

    def opening(self) -> dict:
        return {
            "topic": self.scenario["topic"],
            "level": self.state.cefr_level,
            "opening_line": self.scenario["opening_line"],
            "reply_audio_b64": base64.b64encode(self.opening_wav).decode("ascii"),
        }

    def reset(self) -> dict:
        self.scenario = self.llm.generate_opener(self.state.cefr_level)
        self.history = [{"role": "assistant", "content": self.scenario["opening_line"]}]
        self.opening_wav = self.tts.synthesize(self.scenario["opening_line"])
        log.info("reset: new opener: %s", self.scenario["opening_line"])
        return self.opening()

    def set_mode(self, mode: str) -> dict:
        self.mode = mode
        log.info("mode → %s", mode)
        if mode == "sandbox":
            self.scenario = {"topic": "libre", "setting": "conversation libre", "opening_line": ""}
            self.history = []
            self.opening_wav = b""
            log.info("mode → sandbox: clean slate, user speaks first")
            return {"topic": "libre", "level": self.state.cefr_level}
        else:
            self.scenario = self.llm.generate_opener(self.state.cefr_level)
            self.history = [{"role": "assistant", "content": self.scenario["opening_line"]}]
            self.opening_wav = self.tts.synthesize(self.scenario["opening_line"])
            log.info("mode → guided, new opener: %s", self.scenario["opening_line"])
        return self.opening()

    def set_level(self, level: str) -> dict:
        if level not in VALID_LEVELS:
            raise ValueError(f"invalid level {level!r}")
        self.state.cefr_level = level
        save(self.state, self.state_path)
        if self.mode == "sandbox":
            self.scenario = {"topic": "libre", "setting": "conversation libre", "opening_line": ""}
            self.history = []
            self.opening_wav = b""
            log.info("level → %s (sandbox: no opener)", level)
            return {"topic": "libre", "level": level}
        self.scenario = self.llm.generate_opener(level)
        self.history = [{"role": "assistant", "content": self.scenario["opening_line"]}]
        self.opening_wav = self.tts.synthesize(self.scenario["opening_line"])
        log.info("level → %s, new opener: %s", level, self.scenario["opening_line"])
        return self.opening()


def transcribe_audio_bytes(
    asr: ASREngine, blob: bytes,
    en_asr: EnglishASREngine | None = None,
    level: str = "",
    dual_threshold: float = 0.75,
) -> tuple[str, float, str | None, bool]:
    """Decode browser MediaRecorder blob (webm/opus) → 16 kHz mono float32 → ASR.

    Returns (transcript, confidence, en_context, en_dominant).
    en_context: English ASR text whenever it's non-empty — passed to the LLM as
    auxiliary context so it can interpret code-switched speech.
    en_dominant: True only when English clearly won (lang_prob > 0.9) — replaces
    the French transcript and triggers the A1 English-mode override in the prompt.
    """
    import librosa
    Path("/tmp/last_turn.webm").write_bytes(blob)
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
        input=blob, capture_output=True, check=True,
    )
    Path("/tmp/last_turn.wav").write_bytes(proc.stdout)
    audio, _ = librosa.load(io.BytesIO(proc.stdout), sr=16000, mono=True)
    text, conf = asr.transcribe(audio)
    log.info("audio: %.2fs  conf=%.2f  text=%r", len(audio) / 16000, conf, text)
    en_context = None
    en_dominant = False
    if en_asr and (level == "A1" or (level == "A2" and conf < dual_threshold)):
        en_raw, en_conf = en_asr.transcribe(audio)
        log.info("en_asr: lang_prob=%.2f  text=%r", en_conf, en_raw)
        if en_raw:
            en_context = en_raw
            if en_conf > 0.9:
                log.info("English wins (en=%.2f > fr=%.2f) — using English transcript", en_conf, conf)
                text = en_raw
                en_dominant = True
    return text, conf, en_context, en_dominant


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet default access log
            log.debug(fmt, *args)

        def _send_json(self, status: int, payload: dict):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, rel: str):
            p = STATIC_DIR / rel
            if not p.is_file() or not p.resolve().is_relative_to(STATIC_DIR.resolve()):
                self.send_error(404); return
            data = p.read_bytes()
            ext = p.suffix.lower()
            ctype = {"html": "text/html; charset=utf-8", "js": "application/javascript",
                     "css": "text/css", "ico": "image/x-icon"}.get(ext.lstrip("."), "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send_file("index.html")
            elif self.path == "/opening":
                self._send_json(200, app.opening())
            elif self.path.startswith("/static/"):
                self._send_file(self.path[len("/static/"):])
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            ctype = self.headers.get("Content-Type", "")
            try:
                if self.path == "/level":
                    level = json.loads(body)["level"]
                    self._send_json(200, app.set_level(level))
                    return
                if self.path == "/reset":
                    self._send_json(200, app.reset())
                    return
                if self.path == "/mode":
                    self._send_json(200, app.set_mode(json.loads(body)["mode"]))
                    return
                if self.path != "/turn":
                    self.send_error(404); return
                en_context = None
                en_dominant = False
                if ctype.startswith("application/json"):
                    transcript = json.loads(body)["text"]
                    confidence = 1.0  # typed text bypasses the threshold
                elif ctype.startswith("audio/"):
                    transcript, confidence, en_context, en_dominant = transcribe_audio_bytes(
                        app.asr, body,
                        en_asr=app.en_asr,
                        level=app.state.cefr_level,
                        dual_threshold=app.dual_asr_threshold,
                    )
                else:
                    self._send_json(400, {"error": f"unsupported content-type {ctype!r}"}); return
                if not transcript.strip() and ctype.startswith("application/json"):
                    self._send_json(200, {"transcript": "", "reply": "", "errors": []}); return
                self._send_json(200, app.turn(transcript, confidence, en_context, en_dominant))
            except Exception as e:
                log.exception("POST %s failed", self.path)
                self._send_json(500, {"error": str(e)})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tutor.yaml", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    setup_logging()
    cfg = load_yaml(args.config)
    app = App(cfg)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    log.info("Tutor on http://%s:%d", args.host, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
