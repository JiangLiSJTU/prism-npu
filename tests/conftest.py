"""pytest fixtures shared across the test suite."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SIM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM_ROOT / "src"))


@pytest.fixture(scope="session")
def sim_root() -> Path:
    return SIM_ROOT


@pytest.fixture(scope="session")
def pipe_baseline_path(sim_root: Path) -> Path:
    return sim_root / "data" / "calibration" / "pipe_baseline_per_model.json"


@pytest.fixture(scope="session")
def pipe_baseline(pipe_baseline_path: Path) -> dict:
    if not pipe_baseline_path.is_file():
        pytest.skip(f"baseline JSON not found: {pipe_baseline_path}")
    with pipe_baseline_path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def eta_fit_path(sim_root: Path) -> Path:
    return sim_root / "data" / "calibration" / "eta_physics_fit.json"


@pytest.fixture(scope="session")
def eta_fit(eta_fit_path: Path) -> dict:
    if not eta_fit_path.is_file():
        pytest.skip(f"eta fit JSON not found: {eta_fit_path}")
    with eta_fit_path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def arch_baseline_path(sim_root: Path) -> Path:
    return sim_root / "arch" / "ascend_910b4_for_sweep_v2.yaml"


@pytest.fixture(scope="session")
def pipe_dest_bw_path(sim_root: Path) -> Path:
    return sim_root / "data" / "calibration" / "pipe_dest_bw.json"


@pytest.fixture(scope="session")
def pipe_dest_bw(pipe_dest_bw_path: Path) -> dict:
    """Issue #7 per-config gm_frac calibration (aic_fixpipe / aiv_mte3)."""
    if not pipe_dest_bw_path.is_file():
        pytest.skip(f"pipe_dest_bw JSON not found: {pipe_dest_bw_path}")
    with pipe_dest_bw_path.open(encoding="utf-8") as f:
        return json.load(f)
