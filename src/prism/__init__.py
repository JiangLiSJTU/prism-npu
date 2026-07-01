"""prism package — root."""

import os
from pathlib import Path

__version__ = "0.1.0"


def find_data_root() -> Path:
    """Resolve the repo data root containing arch/, models/, data/.

    Resolution order (first match wins):
      1. ``PRISM_ROOT`` env var (explicit override; CI / tests / pip-install users)
      2. The current working directory if it contains ``data/calibration/``
      3. Climb from this file's location (works for editable installs ``pip install -e .``)

    Raises FileNotFoundError if none of the candidates resolves to a valid layout.
    """
    candidates = []

    env_root = os.environ.get("PRISM_ROOT")
    if env_root:
        candidates.append(Path(env_root).resolve())

    candidates.append(Path.cwd().resolve())

    # Editable-install fallback: src/prism/__init__.py → repo root is parents[2]
    candidates.append(Path(__file__).resolve().parents[2])

    for c in candidates:
        if (c / "data" / "calibration").is_dir() and (c / "arch").is_dir():
            return c

    raise FileNotFoundError(
        "Could not locate prism data root. Set PRISM_ROOT env var to the "
        "directory containing arch/, models/, data/. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def data_root_or_fallback() -> Path:
    """Like find_data_root() but returns the editable-install path on miss
    (avoids import-time crash; modules can still operate if user later
    sets PRISM_ROOT or calls find_data_root() explicitly)."""
    try:
        return find_data_root()
    except FileNotFoundError:
        return Path(__file__).resolve().parents[2]
