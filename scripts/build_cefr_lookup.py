"""Build data/curriculum/cefr_lookup.json from FLELex + Lexique383.

Downloads:
  FLELex (Beacco variant)  — github.com/nathanschulz/french-vocab-tool
  Lexique383               — lexique.org (CC BY SA 4.0)

Output schema:
  {
    "source": "...",
    "words": {
      "<lemma>": {"level": "A1", "pos": "NOM", "gender": "f"},
      ...
    }
  }

Usage:
    uv run python scripts/build_cefr_lookup.py --out data/curriculum/cefr_lookup.json
"""

import argparse
import csv
import io
import json
import os
import sys
import urllib.request
import zipfile

FLELEX_URL = (
    "https://raw.githubusercontent.com/nathanschulz/french-vocab-tool"
    "/main/data/FleLex_TT_Beacco.tsv"
)
LEXIQUE_URL = "http://www.lexique.org/databases/Lexique383/Lexique383.zip"


def fetch(url: str, desc: str) -> bytes:
    print(f"  Downloading {desc}...", end=" ", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    print(f"{len(data)//1024} KB")
    return data


def load_flelex(data: bytes) -> dict:
    result = {}
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t")
    for row in reader:
        word = row["word"].strip().lower()
        result[word] = {
            "level": row["level"].strip(),
            "pos": row["tag"].strip(),
        }
    return result


def load_lexique_gender(data: bytes) -> dict:
    """Return lemma → gender ('m'/'f') using any inflection that has a genre."""
    gender: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open("Lexique383.tsv") as f:
            text = f.read().decode("utf-8")
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    for row in reader:
        lemme = row["lemme"].strip().lower()
        if row["cgram"].strip() == "NOM" and row["genre"].strip() in ("m", "f"):
            if lemme not in gender:
                gender[lemme] = row["genre"].strip()
    return gender


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/curriculum/cefr_lookup.json")
    args = parser.parse_args()

    print("Building CEFR lookup...")

    flelex_bytes = fetch(FLELEX_URL, "FLELex Beacco")
    lexique_bytes = fetch(LEXIQUE_URL, "Lexique383")

    words = load_flelex(flelex_bytes)
    gender = load_lexique_gender(lexique_bytes)

    nouns = sum(1 for e in words.values() if e["pos"] == "NOM")
    for word, entry in words.items():
        if entry["pos"] == "NOM" and word in gender:
            entry["gender"] = gender[word]

    has_gender = sum(1 for e in words.values() if "gender" in e)
    print(f"  {len(words)} words: {nouns} nouns, {has_gender} with gender ({100*has_gender/nouns:.0f}%)")

    from collections import Counter
    levels = Counter(e["level"] for e in words.values())
    print("  Level counts:", dict(sorted(levels.items())))

    output = {
        "source": (
            "FLELex Beacco (openlexicon.fr / nathanschulz/french-vocab-tool) "
            "+ Lexique383 gender (lexique.org). CC BY SA 4.0."
        ),
        "note": "word → {level: A1–C2, pos: NOM|VER|ADJ|ADV|…, gender: m|f (nouns only where available)}",
        "words": words,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  Wrote {args.out} ({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    main()
