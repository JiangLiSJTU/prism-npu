#!/usr/bin/env python3
"""Thin CLI wrapper for prism.ceiling.predict:main.

Two ways to run:
  1) After `pip install -e .`: `prism-ceiling <args>`
  2) From source without install: `python3 scripts/prism_ceiling.py <args>`
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prism.ceiling.predict import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
