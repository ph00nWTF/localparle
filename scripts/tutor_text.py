"""Text-mode French tutor — stdin/stdout, no ASR/TTS. Demo-able while HPC trains.

    uv run python scripts/tutor_text.py --config configs/tutor.yaml
    uv run python scripts/tutor_text.py --config configs/tutor.yaml --ask-level
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from french_tutor.config import load_yaml
from french_tutor.learner_state import VALID_LEVELS, LearnerState, load, save
from french_tutor.pedagogy import update_after_turn
from french_tutor.prompt_builder import build_turn_prompt, load_cefr_lookup
from french_tutor.tutor import LLMEngine
from french_tutor.utils import setup_logging


def ask_level() -> str:
    while True:
        choice = input("Quel est ton niveau ? [A1/A2/B1/B2/C1/C2] ").strip().upper()
        if choice in VALID_LEVELS:
            return choice
        print(f"  niveau invalide — choisis parmi {', '.join(VALID_LEVELS)}", file=sys.stderr)


def get_or_create_state(state_path: Path, *, force_ask: bool, default_level: str) -> LearnerState:
    state = load(state_path)
    if state is not None and not force_ask:
        return state
    if state is None:
        print(f"Aucun état trouvé à {state_path}. Bienvenue !")
    level = ask_level() if (state is None or force_ask) else default_level
    return state if state and not force_ask else LearnerState.default(level)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tutor.yaml", type=Path)
    parser.add_argument("--ask-level", action="store_true", help="Re-prompt for CEFR level even if state exists")
    args = parser.parse_args()

    setup_logging()
    cfg = load_yaml(args.config)

    state_path = Path(cfg["pedagogy"]["learner_state_path"])
    state = get_or_create_state(
        state_path,
        force_ask=args.ask_level,
        default_level=cfg["pedagogy"]["default_level"],
    )
    save(state, state_path)

    cefr_lookup = load_cefr_lookup()
    llm = LLMEngine(
        ollama_url=cfg["llm"]["ollama_url"],
        model=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
    )
    scenario = llm.generate_opener(state.cefr_level)

    print(f"\n--- Niveau {state.cefr_level} | sujet : {scenario['topic']} ---")
    print(f"Tuteur : {scenario['opening_line']}\n")
    history: list[dict] = [{"role": "assistant", "content": scenario["opening_line"]}]

    while True:
        try:
            transcript = input("Toi : ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not transcript:
            continue
        if transcript.lower() in ("/quit", "/exit"):
            break

        now = datetime.now(UTC)
        system, user, schema = build_turn_prompt(
            state, scenario, history, transcript,
            cefr_lookup=cefr_lookup, now=now,
        )
        try:
            result = llm.respond(system, user, history, schema)
        except Exception as e:
            print(f"\nErreur LLM : {e}\nVérifie qu'Ollama tourne et que '{cfg['llm']['model']}' est installé "
                  f"(make pull-mistral).", file=sys.stderr)
            return 1

        update_after_turn(state, transcript, result.get("errors", []), now=now)
        save(state, state_path)
        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": result["reply"]})

        print(f"Tuteur : {result['reply']}")
        if result.get("errors"):
            words = ", ".join(e["word"] for e in result["errors"])
            print(f"  (erreurs notées : {words})")
        print()

    print(f"\nFin de session. {state.turn_count} tour(s). État : {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
