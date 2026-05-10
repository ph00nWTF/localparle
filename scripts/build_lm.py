"""Build a 4-gram KenLM French language model from Common Voice train transcripts.

    uv run python scripts/build_lm.py \
        --processed-dir /kuhpc/scratch/deng/e394n539/french-data/processed \
        --out-dir /kuhpc/scratch/deng/e394n539/french-data/lm \
        --lmplz /kuhpc/scratch/deng/e394n539/kenlm_build/build/bin/lmplz \
        --build-binary /kuhpc/scratch/deng/e394n539/kenlm_build/build/bin/build_binary

Pipeline:
1. Read all sentences from train split (already-normalized text labels).
2. Re-normalize via french_tutor.data.normalize_transcript so corpus matches model vocab.
3. Write one sentence per line to corpus.txt.
4. Run lmplz -o 4 to produce ARPA-format LM.
5. Run build_binary to convert ARPA -> .bin (faster load, smaller).
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path

from datasets import load_from_disk

from french_tutor.data import normalize_transcript
from french_tutor.utils import setup_logging

log = logging.getLogger("build_lm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--lmplz", required=True, type=Path,
                        help="Path to KenLM lmplz binary")
    parser.add_argument("--build-binary", required=True, type=Path,
                        help="Path to KenLM build_binary tool")
    parser.add_argument("--order", default=4, type=int, help="N-gram order (default 4)")
    args = parser.parse_args()

    setup_logging()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.processed_dir / "train"
    log.info("loading train transcripts from %s", train_path)
    ds = load_from_disk(str(train_path))
    sentences = ds["sentence"]
    log.info("got %d sentences", len(sentences))

    corpus_txt = args.out_dir / "corpus.txt"
    log.info("writing normalized corpus to %s", corpus_txt)
    n_written = 0
    n_skipped = 0
    with corpus_txt.open("w", encoding="utf-8") as f:
        for s in sentences:
            norm = normalize_transcript(s)
            if norm:
                f.write(norm + "\n")
                n_written += 1
            else:
                n_skipped += 1
    log.info("wrote %d sentences (skipped %d empty after normalize)", n_written, n_skipped)

    tmp_dir = args.out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    arpa = args.out_dir / f"fr_{args.order}gram.arpa"
    log.info("running lmplz -o %d -> %s (memory cap 8G, tmp=%s)", args.order, arpa, tmp_dir)
    t0 = time.time()
    with corpus_txt.open("r") as inp, arpa.open("w") as out:
        proc = subprocess.run(
            [str(args.lmplz), "-o", str(args.order), "-S", "8G",
             "-T", str(tmp_dir), "--discount_fallback"],
            stdin=inp, stdout=out, check=True,
        )
    log.info("lmplz finished in %.1fs (rc=%d)", time.time() - t0, proc.returncode)

    binlm = args.out_dir / f"fr_{args.order}gram.bin"
    log.info("converting to binary: %s -> %s", arpa, binlm)
    t0 = time.time()
    subprocess.run([str(args.build_binary), str(arpa), str(binlm)], check=True)
    log.info("build_binary finished in %.1fs", time.time() - t0)

    log.info("done. ARPA=%s  BIN=%s", arpa, binlm)
    log.info("ARPA size: %.1f MB", arpa.stat().st_size / 1024 / 1024)
    log.info("BIN size:  %.1f MB", binlm.stat().st_size / 1024 / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
