#!/usr/bin/env bash
# Build a French 4-gram KenLM trained on CV transcripts + French Wikipedia.
#
# Prerequisites (run once):
#   sudo apt install -y cmake libboost-all-dev libeigen3-dev build-essential
#   pip install wikiextractor   (or: uv add wikiextractor --dev)
#
# Usage:
#   bash scripts/build_wiki_lm.sh
#
# Output:
#   models/lm/lm_wiki.binary   — drop-in replacement for lm.binary
#   models/lm/unigrams_wiki.txt
#
# After it finishes, update configs/tutor.yaml:
#   lm_path: models/lm/lm_wiki.binary
#   unigrams_path: models/lm/unigrams_wiki.txt
# Then re-run scripts/tune_lm.py with the new LM to find new best alpha/beta.

set -euo pipefail

WIKI_DUMP_URL="https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles.xml.bz2"
SCRATCH_DIR="/mnt/d/french-lm-scratch"
KENLM_BUILD="$SCRATCH_DIR/kenlm/build"
CORPUS="$SCRATCH_DIR/corpus.txt"
ARPA="$SCRATCH_DIR/lm_wiki.arpa"
OUT_DIR="models/lm"

mkdir -p "$SCRATCH_DIR" "$OUT_DIR"

# ── 1. Build kenlm (skip if already built) ───────────────────────────────────
if [ ! -f "$KENLM_BUILD/bin/lmplz" ]; then
    echo "==> Building kenlm from source…"
    cd "$SCRATCH_DIR"
    git clone --depth 1 https://github.com/kpu/kenlm.git
    mkdir -p kenlm/build && cd kenlm/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j"$(nproc)"
    cd -
    echo "==> kenlm built."
else
    echo "==> kenlm already built, skipping."
fi

LMPLZ="$KENLM_BUILD/bin/lmplz"
BUILD_BINARY="$KENLM_BUILD/bin/build_binary"

# ── 2. Download French Wikipedia dump (skip if already downloaded) ────────────
DUMP="$SCRATCH_DIR/frwiki-latest.xml.bz2"
if [ ! -f "$DUMP" ]; then
    echo "==> Downloading French Wikipedia (~5 GB compressed)…"
    wget -c "$WIKI_DUMP_URL" -O "$DUMP"
else
    echo "==> Wikipedia dump already downloaded, skipping."
fi

# ── 3. Extract plain text ─────────────────────────────────────────────────────
WIKI_TEXT="$SCRATCH_DIR/wiki_text"
if [ ! -d "$WIKI_TEXT" ]; then
    echo "==> Extracting Wikipedia text (wikiextractor)…"
    uv run python -m wikiextractor.WikiExtractor "$DUMP" --no-templates -o "$WIKI_TEXT" -q
else
    echo "==> Wikipedia text already extracted, skipping."
fi

# ── 4. Build corpus (CV transcripts + Wikipedia) ─────────────────────────────
echo "==> Building combined corpus…"
# CV train transcripts (already on disk as TSV)
CV_TSV="data/mdc-extracted/cv-corpus-25.0-2026-03-09/fr/train.tsv"
if [ -f "$CV_TSV" ]; then
    tail -n +2 "$CV_TSV" | cut -f3 | tr '[:upper:]' '[:lower:]' > "$CORPUS"
    echo "  CV transcripts: $(wc -l < "$CORPUS") lines"
else
    echo "  WARNING: CV train.tsv not found at $CV_TSV — using Wikipedia only"
    > "$CORPUS"
fi
# Wikipedia plain text (one paragraph per line)
find "$WIKI_TEXT" -name "wiki_*" -exec grep -v '^<' {} \; \
    | tr '[:upper:]' '[:lower:]' \
    | grep -v '^$' >> "$CORPUS"
echo "  Combined corpus: $(wc -l < "$CORPUS") lines"

# ── 5. Train 4-gram KenLM ─────────────────────────────────────────────────────
echo "==> Training 4-gram LM (lmplz)… this takes ~30-60 min"
# --memory 4G caps RAM so WSL2 doesn't OOM; uses disk temp files instead
head -5000000 "$CORPUS" > "$SCRATCH_DIR/corpus_small.txt"
mkdir -p /home/phoon/kenlm_tmp
"$LMPLZ" -o 3 --discount_fallback --prune 0 0 1 --memory 4G -T /home/phoon/kenlm_tmp < "$SCRATCH_DIR/corpus_small.txt" > "$ARPA"
echo "==> ARPA written: $ARPA"

# ── 6. Build binary + extract unigrams ───────────────────────────────────────
echo "==> Building binary…"
"$BUILD_BINARY" trie "$ARPA" "$OUT_DIR/lm_wiki.binary"

echo "==> Extracting unigrams…"
uv run python - <<'PYEOF'
import re, sys
unigrams = []
with open(sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/french-lm-scratch/lm_wiki.arpa") as f:
    in_1grams = False
    for line in f:
        if line.strip() == "\\1-grams:":
            in_1grams = True; continue
        if line.startswith("\\") and in_1grams:
            break
        if in_1grams and line.strip():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                unigrams.append(parts[1])
with open("models/lm/unigrams_wiki.txt", "w") as f:
    f.write("\n".join(unigrams))
print(f"Unigrams written: {len(unigrams)}")
PYEOF

echo ""
echo "==> Done! New LM at models/lm/lm_wiki.binary"
echo "==> Update configs/tutor.yaml:"
echo "      lm_path: models/lm/lm_wiki.binary"
echo "      unigrams_path: models/lm/unigrams_wiki.txt"
echo "==> Then re-run: uv run python scripts/tune_lm.py"
