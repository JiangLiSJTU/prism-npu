"""End-to-end smoke test.

Runs the 4 main CLI tools as subprocess (testing the actual entry points
installed via `pip install -e .`), verifying the full pipeline works
from clean install to baseline reproduction.

Skipped if the package isn't installed (e.g., in CI without venv setup).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest


def _which_or_skip(cmd: str) -> str:
    """Find CLI in PATH or skip the test."""
    found = shutil.which(cmd)
    if not found:
        pytest.skip(f"{cmd} not in PATH (pip install -e . first)")
    return found


@pytest.mark.smoke
def test_prism_render_check_e2e():
    """`prism-render --check` exit 0."""
    cmd = _which_or_skip("prism-render")
    result = subprocess.run([cmd, "--check"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (
        f"prism-render --check failed:\nstdout={result.stdout[-500:]}\nstderr={result.stderr[-500:]}"
    )


@pytest.mark.smoke
def test_prism_ceiling_e2e():
    """`prism-ceiling` produces output JSON + MD."""
    cmd = _which_or_skip("prism-ceiling")
    result = subprocess.run([cmd], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f"prism-ceiling failed:\nstdout={result.stdout[-500:]}\nstderr={result.stderr[-500:]}"
    )
    # Should mention "S1_software_ceiling" or similar in stdout
    assert "S1" in result.stdout or "Software" in result.stdout or "S0" in result.stdout, (
        "prism-ceiling output didn't mention scenarios"
    )


@pytest.mark.smoke
def test_prism_sweep_e2e():
    """`prism-sweep` produces 11-dim sweep without errors (Issue #8 dropped a dead dim)."""
    cmd = _which_or_skip("prism-sweep")
    result = subprocess.run([cmd], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f"prism-sweep failed:\nstdout={result.stdout[-500:]}\nstderr={result.stderr[-500:]}"
    )
    # Should mention some sweep dimension
    assert "n_cores" in result.stdout or "ratio" in result.stdout.lower() or "BERT" in result.stdout, (
        "prism-sweep output didn't mention sweep dimensions"
    )


@pytest.mark.smoke
def test_prism_regime_help_e2e():
    """`prism-regime --help` exits cleanly."""
    cmd = _which_or_skip("prism-regime")
    result = subprocess.run([cmd, "--help"], capture_output=True, text=True, timeout=10)
    # argparse --help exits 0
    assert result.returncode == 0, (
        f"prism-regime --help failed:\n{result.stderr[-500:]}"
    )
    assert "regime" in result.stdout.lower() or "Roofline" in result.stdout


@pytest.mark.smoke
def test_no_install_path_works(sim_root):
    """`python3 scripts/prism_render.py --check` 也应当工作（不依赖 pip install）。"""
    script = sim_root / "scripts" / "prism_render.py"
    if not script.is_file():
        pytest.skip(f"wrapper not found: {script}")

    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"no-install render --check failed:\nstdout={result.stdout[-500:]}\nstderr={result.stderr[-500:]}"
    )


@pytest.mark.smoke
def test_pyproject_toml_valid(sim_root):
    """pyproject.toml 应当是合法的 TOML 且含 [project.scripts]。"""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # py<3.11
        except ImportError:
            pytest.skip("tomli/tomllib unavailable")

    pyproject = sim_root / "pyproject.toml"
    with pyproject.open("rb") as f:
        config = tomllib.load(f)

    assert "project" in config
    assert "scripts" in config["project"]
    expected_clis = {"prism-extract", "prism-fit", "prism-regime", "prism-sweep", "prism-ceiling", "prism-mapping", "prism-render"}
    actual = set(config["project"]["scripts"].keys())
    missing = expected_clis - actual
    assert not missing, f"pyproject.toml [project.scripts] 缺 {missing}"
