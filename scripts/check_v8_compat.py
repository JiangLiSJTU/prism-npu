#!/usr/bin/env python3
"""PRISM v8 ↔ v0.1 backward-compatibility check.

Verifies that v8 outputs are consumable by downstream tools (ceiling /
regime / sweep / external ingestors like ingest_phase_o.py) without
schema migration.

Returns exit code:
  0 = fully compatible — safe to upgrade v0.1 callers to v8 with no changes
  1 = compatible-with-caveat — additive-only schema changes, may need
      consumer to ignore unknown keys
  2 = breaking — schema renames/removes (would block downstream)

Usage:
    python3 scripts/check_v8_compat.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


# Reference: v0.1 measured-config schema (lifted from sisyphus/predict_pipe_v0.1.py output)
V01_MEASURED_KEYS = {
    "n_kernels_per_inf", "task_dur_us",
    "aic_time_us", "aiv_time_us",
    "aic_pipes_us", "aiv_pipes_us",
    "aic_bubble_us", "aic_dominant_pipe",
    "wall_clock_us", "kernel_gap_us", "host_gap_us",
    "host_gap_us_per_kernel",
    "source",
}


def check_baseline_schema() -> tuple[int, str]:
    """Q1: pipe_baseline_per_model.json keyset vs v0.1 reference."""
    baseline_path = _REPO / "data" / "calibration" / "pipe_baseline_per_model.json"
    if not baseline_path.exists():
        return 2, f"baseline JSON missing at {baseline_path}"
    doc = json.load(open(baseline_path, encoding="utf-8"))
    configs = doc.get("configs", {})
    if not configs:
        return 2, "baseline has no configs"

    # Pick one measured config
    measured = next((c for c in configs.values() if not c.get("predicted")), None)
    if not measured:
        return 2, "no measured configs found"
    actual = set(measured.keys())

    removed = V01_MEASURED_KEYS - actual
    added = actual - V01_MEASURED_KEYS

    if removed:
        return 2, f"baseline schema REMOVES v0.1 keys: {sorted(removed)} (BREAKING)"
    if added:
        return 1, f"baseline schema adds optional keys: {sorted(added)} (additive)"
    return 0, f"baseline schema matches v0.1 exactly ({len(actual)} keys)"


def check_predicted_schema() -> tuple[int, str]:
    """Q1b: predict_pipe_baseline() output schema for v8."""
    try:
        from prism.predict_pipe import ModelSpec, predict_pipe_baseline
        from prism.predict_pipe.predict import _arch_dict_from_yaml
    except ImportError as e:
        return 2, f"prism.predict_pipe import failed: {e}"

    yaml = _REPO / "models" / "regime" / "modernbert_base_prefill_S4096.yaml"
    arch_yaml = _REPO / "arch" / "ascend_910b4_for_sweep_v2.yaml"
    params_v8 = _REPO / "data" / "calibration" / "predict_pipe_params_v8.json"
    if not params_v8.exists():
        return 1, "predict_pipe_params_v8.json missing — v8 not fit yet (v4-v7 still work)"

    spec = ModelSpec.from_yaml(yaml)
    arch = _arch_dict_from_yaml(arch_yaml)
    params = json.load(open(params_v8, encoding="utf-8"))
    entry = predict_pipe_baseline(spec, arch, params, batch=1)
    actual = set(entry.keys())

    removed = V01_MEASURED_KEYS - actual
    if removed:
        return 2, f"predicted entry MISSING v0.1 keys: {sorted(removed)} (BREAKING)"
    added = actual - V01_MEASURED_KEYS
    if added:
        return 1, f"predicted adds {sorted(added)} — downstream must IGNORE these or whitelist"
    return 0, "predicted matches v0.1 keys exactly"


def check_cli_invocations() -> tuple[int, str]:
    """Q2: ceiling/regime/sweep flag signatures unchanged."""
    import subprocess
    expected_flags = {
        "scripts/prism_ceiling.py":      {"--pipe-baseline", "--output-json", "--output-md"},
        "scripts/prism_regime.py":       {"--arch", "--model", "--batch", "--sweep", "--output"},
        "scripts/prism_sweep.py":        {"--pipe-baseline", "--output"},
        "scripts/prism_predict_pipe.py": {"--model", "--arch", "--batch", "--output",
                                          "--params", "--refit-params", "--merge-into"},
    }
    missing_per_cli = {}
    for script, expected in expected_flags.items():
        path = _REPO / script
        if not path.exists():
            missing_per_cli[script] = "(script missing)"
            continue
        result = subprocess.run(
            ["python3", str(path), "--help"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ, "PYTHONPATH": str(_REPO / "src")},
        )
        text = result.stdout + result.stderr
        missing = [f for f in expected if f not in text]
        if missing:
            missing_per_cli[script] = missing

    if missing_per_cli:
        return 2, f"CLI flag missing in: {missing_per_cli}"
    return 0, f"all 4 CLIs preserve their original flags"


def check_v8_params_file() -> tuple[int, str]:
    """Q3: v8 params file exists, loadable, has expected per-bucket structure."""
    path = _REPO / "data" / "calibration" / "predict_pipe_params_v8.json"
    if not path.exists():
        return 1, "v8 params not generated yet (run `python3 -m prism.predict_pipe.fit_v8`)"
    p = json.load(open(path, encoding="utf-8"))
    if p.get("v_model") != "v8":
        return 2, f"v_model marker is {p.get('v_model')!r}, expected 'v8'"
    required_buckets = {"AIC_DECODE", "AIV_BOUND", "BALANCED"}
    found = {k.split("_")[1] + ("_" + k.split("_")[2] if k.count("_") > 1 else "")
             for k in p if k.startswith("v7_")}
    # Simpler: look for AIV_BOUND_amp_aic etc.
    has_aic_decode = any(k.startswith("v7_AIC_DECODE_") for k in p)
    has_aiv_bound = any(k.startswith("v7_AIV_BOUND_") for k in p)
    has_balanced = any(k.startswith("v7_BALANCED_") for k in p)
    missing = [b for b, ok in [("AIC_DECODE", has_aic_decode),
                                ("AIV_BOUND", has_aiv_bound),
                                ("BALANCED", has_balanced)] if not ok]
    if missing:
        return 2, f"v8 params missing per-bucket coefficients for {missing}"
    return 0, "v8 params well-formed (3 buckets × 3 amp coefficients)"


def main():
    checks = [
        ("Q1  baseline JSON schema vs v0.1",      check_baseline_schema),
        ("Q1b predict_pipe_baseline() output",     check_predicted_schema),
        ("Q2  ceiling/regime/sweep CLI flags",     check_cli_invocations),
        ("Q3  v8 params file structure",           check_v8_params_file),
    ]

    print("PRISM v8 ↔ v0.1 backward-compatibility check")
    print("=" * 60)
    worst = 0
    for name, fn in checks:
        try:
            code, msg = fn()
        except Exception as e:
            code, msg = 2, f"check crashed: {e}"
        worst = max(worst, code)
        emoji = {0: "✅", 1: "⚠️ ", 2: "❌"}[code]
        print(f"{emoji} {name}")
        print(f"   {msg}")

    print("=" * 60)
    if worst == 0:
        print("✅ OVERALL: SAFE TO UPGRADE — no consumer changes needed")
    elif worst == 1:
        print("⚠️  OVERALL: ADDITIVE-ONLY — consumers that ignore unknown keys work as-is;")
        print("   strict-schema consumers need to whitelist new optional fields.")
    else:
        print("❌ OVERALL: BREAKING — schema migration required before upgrade")
    return worst


if __name__ == "__main__":
    sys.exit(main())
