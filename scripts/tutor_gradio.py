"""Gradio tutor entrypoint — browser UI at localhost:7860."""
from __future__ import annotations

import argparse
from pathlib import Path

from french_tutor.apps import run_gradio
from french_tutor.config import load_yaml
from french_tutor.utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tutor.yaml", type=Path)
    args = parser.parse_args()

    setup_logging()
    run_gradio(load_yaml(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
