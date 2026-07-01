#!/usr/bin/env python3
"""Thin CLI wrapper for prism.predict_pipe.predict:main.

Equivalent to the ``prism-predict-pipe`` entry point installed by
``pip install -e .``; usable without install by adding src/ to PYTHONPATH.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prism.predict_pipe.predict import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main() or 0)
