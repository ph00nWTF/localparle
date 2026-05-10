"""Build the system + user prompt and JSON schema for a single turn.

The orchestrator passes the result to LLMEngine.respond. The LLM returns
{"reply": str, "errors": [...]} per the schema, and the orchestrator updates
learner state with the errors via pedagogy.update_after_turn.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .learner_state import LearnerState
from .pedagogy import english_scaffold_for_level, next_due_cards

_CEFR_LOOKUP_PATH = Path(__file__).parents[2] / "data" / "curriculum" / "cefr_lookup.json"


def load_cefr_lookup() -> dict:
    return json.loads(_CEFR_LOOKUP_PATH.read_text(encoding="utf-8"))["words"]


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {
            "type": "string",
            "description": "What you (the tutor) say next, in French.",
        },
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "error_type": {
                        "type": "string",
                        "enum": ["gender", "agreement", "conjugation", "false_cognate", "other"],
                    },
                    "word": {"type": "string"},
                    "correction": {"type": "string"},
                },
                "required": ["error_type", "word", "correction"],
            },
        },
    },
    "required": ["reply", "errors"],
}


def build_turn_prompt(
    state: LearnerState,
    scenario: dict,
    history: list[dict],
    transcript: str,
    *,
    cefr_lookup: dict,
    now: datetime,
    mode: str = "guided",
    en_context: str | None = None,
    en_dominant: bool = False,
) -> tuple[str, str, dict]:
    """Build (system_prompt, user_prompt, json_schema) for a single turn."""
    scaffold = english_scaffold_for_level(state.cefr_level)
    is_first_turn = state.turn_count == 0
    due = next_due_cards(state, now=now, k=3) if is_first_turn else []
    system = _build_system_prompt(state.cefr_level, scenario, scaffold, due, mode=mode)
    if en_dominant and state.cefr_level == "A1":
        system = (
            "CRITICAL OVERRIDE — the user spoke English. "
            f"Their English message was: \"{en_context}\". "
            "Your `reply` MUST be entirely in English. Do NOT write French sentences in `reply` except for one short example phrase the user can try to repeat. "
            "Say something like: \"It looks like you spoke English — that's okay! At A1 level I can help. "
            "Try saying: [one simple French phrase relevant to the topic]. Repeat after me!\"\n\n"
        ) + system
    user = transcript if not is_first_turn else f"[début de session] {transcript}"
    if en_context:
        user += f"\n[English ASR: {en_context}]"
    return system, user, JSON_SCHEMA


def _build_system_prompt(
    level: str,
    scenario: dict,
    scaffold: str,
    due_cards: list,
    *,
    mode: str = "guided",
) -> str:
    parts: list[str] = []

    if mode == "guided":
        parts.append(
            f"Tu joues un rôle dans une conversation immersive. "
            f"Cadre : {scenario['setting']} "
            f"L'apprenant est au niveau CEFR {level}. "
            f"La conversation est en cours — n'accueille pas l'apprenant à nouveau. "
            f"Lis l'historique et réponds au DERNIER message de l'apprenant."
        )
    else:
        parts.append(
            f"Tu es un tuteur de français bienveillant en mode conversation libre. "
            f"L'apprenant est au niveau CEFR {level}. "
            f"Suis la conversation où qu'elle mène — pas de sujet imposé. "
            f"Lis l'historique et réponds au DERNIER message de l'apprenant."
        )

    parts.append(
        "Réponds DIRECTEMENT au contenu de ce que l'apprenant vient de dire — "
        "montre que tu as compris ce qu'il a dit. "
        "Puis pose une question concrète liée à son message pour faire avancer la conversation. "
        "Trois phrases maximum. "
        "Calibre ton vocabulaire et ta grammaire au niveau i+1 (légèrement au-dessus du niveau "
        "actuel) pour que l'apprenant progresse naturellement (Krashen)."
    )

    if scaffold == "free":
        parts.append(
            "Niveau A1 : pour CHAQUE mot français un peu nouveau, ajoute une courte "
            "traduction anglaise entre parenthèses. "
            "Exemple : « Voulez-vous un croissant (a croissant) avec votre café (coffee) ? » "
            "C'est obligatoire — l'apprenant ne connaît pas encore tout le vocabulaire."
        )
    elif scaffold == "fallback":
        parts.append(
            "Niveau A2 : utilise une traduction anglaise entre parenthèses uniquement "
            "si l'apprenant ne comprend pas après deux tours."
        )
    else:
        parts.append(
            "Réponds en français uniquement. Aucun anglais, aucune traduction. "
            "Toutes les corrections se font en français."
        )

    if scaffold == "free":
        parts.append(
            "IMPORTANT : si la transcription ressemble à de l'anglais ou est inintelligible "
            "(mélange de mots sans sens en français), réponds UNIQUEMENT en anglais : "
            "\"It looks like you spoke English — that's okay! At A1 level I can help. "
            "Try saying: [donne une phrase française simple liée au contexte]. Repeat after me!\". "
            "Ne prétends jamais que l'apprenant a dit quelque chose en français s'il ne l'a pas fait. "
            "Le message peut contenir '[English ASR: ...]' — c'est la transcription anglaise du même audio ; "
            "utilise-la pour comprendre ce que l'apprenant a réellement dit."
        )
        parts.append(
            "Si l'apprenant fait une erreur en français, pose une question d'élicitation courte — "
            "tu PEUX utiliser l'anglais pour aider (niveau A1). "
            "Exemple : s'il dit « le maison », demande « Is 'maison' masculine or feminine? "
            "Le or la ? »"
        )
    elif scaffold == "fallback":
        parts.append(
            "Si l'apprenant fait une erreur, pose une question d'élicitation en français. "
            "Exemple : s'il dit « le maison », demande « le ou la maison ? » "
            "Si l'erreur persiste après deux tours, tu peux expliquer brièvement en anglais."
        )
    else:
        parts.append(
            "Si l'apprenant fait une erreur, pose une question d'élicitation en français uniquement "
            "(pas de recast, pas de correction directe). "
            "Exemple : s'il dit « le maison », demande « le ou la maison ? » "
            "L'objectif est que l'apprenant se corrige lui-même (Lyster & Ranta 1997)."
        )

    if due_cards:
        words = ", ".join(c.word for c in due_cards)
        parts.append(
            f"Au début de cette session, intègre naturellement ces mots à réviser "
            f"dans la conversation : {words}."
        )

    parts.append(
        "Réponds STRICTEMENT en JSON conforme au schéma. "
        '`reply` : ta réponse en français. '
        "`errors` : TOUTES les erreurs de l'apprenant dans son dernier message. "
        "Le champ `correction` contient la forme française correcte (PAS la traduction anglaise). "
        "Si tu corriges quelque chose dans `reply`, tu DOIS l'ajouter dans `errors`. "
        "Exemples :\n"
        "  - apprenant écrit « le maison » → "
        '{"error_type":"gender","word":"maison","correction":"la maison"}\n'
        "  - apprenant écrit « cafe » (sans accent) → "
        '{"error_type":"other","word":"cafe","correction":"café"}\n'
        "  - apprenant écrit « j'ai allé » (mauvais auxiliaire) → "
        '{"error_type":"conjugation","word":"j\'ai allé","correction":"je suis allé"}\n'
        "Types d'erreur : gender, agreement, conjugation, false_cognate, other. "
        "Si l'apprenant n'a fait aucune erreur, laisse `errors` vide."
    )

    return "\n\n".join(parts)
