"""v5 fit (Issue #2 v5 — overfit-safe replacements for v4).

Splits (per `splits.py`):
  TRAIN          9 configs  (b=1 + Qwen3-S512-b4 anchor; w_proxy ∈ [89, 763])
  VAL_batch      9 configs  (b=4/8/16 of same model families)
  VAL_size       3 configs  (Llama / Qwen2.5 / SmolLM2-360M, w_proxy ∈ [708, 2147])

Loss: wall_clock MAE on TRAIN only (no peeking at val).

After fit completes, reports:
  - TRAIN MAE + each-config err
  - VAL_batch MAE (batch extrapolation)
  - VAL_size MAE (size extrapolation — the dimension that broke v4)

Acceptance bar (set by user mandate 2026-05-15):
  - TRAIN MAE ≤ 15%   (v4: 4.9% — accept slight regression to gain generalization)
  - VAL_batch MAE ≤ 30%
  - VAL_size MAE ≤ 30%   (v4: 137-1156% catastrophic; this is the must-improve)

If any val MAE > 30%, formula needs adjustment (NOT just refit).
"""
from __future__ import annotations
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import scipy.optimize

from . import physics_v5
from .model_spec import ModelSpec, KNOWN_MODELS
from .predict import _arch_dict_from_yaml, predict_pipe_baseline
from .splits import TRAIN_CONFIGS, VAL_BATCH_CONFIGS, VAL_SIZE_CONFIGS, resolve_path


def _qwen3_with_S(S: int) -> ModelSpec:
    """Build Qwen3-prefill spec at arbitrary S by cloning the S256 base."""
    import dataclasses
    base = KNOWN_MODELS["Qwen3-prefill-S256-b1"]
    return dataclasses.replace(base, S=S, name=f"Qwen3-prefill-S{S}")


def _load_spec(cfg_name: str, yaml_path: str) -> "ModelSpec | None":
    """Prefer KNOWN_MODELS (no gemm_spec required); fallback to YAML.

    The base cfg_name maps directly into KNOWN_MODELS, but for batch>1 we
    look up the b=1 variant (same model geometry, batch is passed separately).
    """
    if cfg_name in KNOWN_MODELS:
        return KNOWN_MODELS[cfg_name]
    base_name = cfg_name.rsplit("-b", 1)[0] + "-b1"
    if base_name in KNOWN_MODELS:
        return KNOWN_MODELS[base_name]
    # Special: Qwen3-prefill-S{S}-b{B} — clone S256 base with new S
    if cfg_name.startswith("Qwen3-prefill-S"):
        try:
            S_str = cfg_name.split("-S")[1].split("-")[0]
            return _qwen3_with_S(int(S_str))
        except (IndexError, ValueError):
            pass
    try:
        return ModelSpec.from_yaml(resolve_path(yaml_path))
    except (KeyError, FileNotFoundError) as e:
        print(f"  [skip] {cfg_name}: {e}")
        return None


# Parameters to fit (9 free). Fixed: K0, H_*, decode constants, aiv_C_kernel, aiv_C_data
_FIT_KEYS = (
    "aic_amp_alpha", "aic_amp_max",
    "aiv_amp_a0", "aiv_amp_a1", "aiv_amp_a2", "aiv_amp_W_sat",
    "nk_mult_base", "nk_mult_max", "nk_W_sat",
)


def _build_dataset(configs, baseline_doc, arch):
    """Materialize (spec, batch, measured_wall) tuples; skip missing measurements."""
    dataset = []
    for cfg_name, yaml_path, batch in configs:
        if cfg_name not in baseline_doc["configs"]:
            continue
        meas = baseline_doc["configs"][cfg_name]
        if not meas.get("wall_clock_us"):
            continue
        spec = _load_spec(cfg_name, yaml_path)
        if spec is None:
            continue
        dataset.append((cfg_name, spec, batch, meas))
    return dataset


def _eval_one(spec, arch, batch, params, meas) -> Dict[str, float]:
    pred = predict_pipe_baseline(spec, arch, params, batch=batch)
    pw, mw = pred["wall_clock_us"], meas["wall_clock_us"]
    return {
        "wall_err_pct": abs(pw - mw) / mw * 100 if mw else 0.0,
        "aic_err_pct": abs(pred["aic_time_us"] - meas["aic_time_us"]) /
                       max(meas["aic_time_us"], 1) * 100,
        "aiv_err_pct": abs(pred["aiv_time_us"] - meas["aiv_time_us"]) /
                       max(meas["aiv_time_us"], 1) * 100,
        "nk_err_pct": abs(pred["n_kernels_per_inf"] - meas["n_kernels_per_inf"]) /
                      max(meas["n_kernels_per_inf"], 1) * 100,
        "wall_pred": pw, "wall_meas": mw,
    }


def _mae(dataset, arch, params) -> float:
    errs = []
    for cfg_name, spec, batch, meas in dataset:
        r = _eval_one(spec, arch, batch, params, meas)
        errs.append(r["wall_err_pct"])
    return statistics.mean(errs) if errs else 0.0


def _make_params(theta: List[float], fixed: Dict[str, float]) -> Dict[str, float]:
    """Build a fitted_params dict from a fit-vector + fixed scalars."""
    p = dict(fixed)
    p["v_model"] = "v5"
    for key, val in zip(_FIT_KEYS, theta):
        p[key] = val
    # Decode constants — not fit (only 1 decode config in train)
    p["aic_amp_decode"] = float(fixed.get("aic_amp_decode", 0.85))
    p["nk_mult_decode"] = float(fixed.get("nk_mult_decode", 4.7))
    return p


def fit_v5(baseline_path: Path | str,
           arch_yaml: Path | str,
           v4_params_path: Path | str) -> Dict:
    """Fit v5 params on TRAIN; report TRAIN/VAL_batch/VAL_size MAE.

    Returns a dict ready to dump to predict_pipe_params.json.
    """
    baseline = json.load(open(baseline_path, encoding="utf-8"))
    arch = _arch_dict_from_yaml(arch_yaml)
    v4_params = json.load(open(v4_params_path, encoding="utf-8"))   # for K0, H_prefill, H_decode

    train = _build_dataset(TRAIN_CONFIGS, baseline, arch)
    val_b = _build_dataset(VAL_BATCH_CONFIGS, baseline, arch)
    val_s = _build_dataset(VAL_SIZE_CONFIGS, baseline, arch)
    print(f"Datasets: TRAIN={len(train)}  VAL_batch={len(val_b)}  VAL_size={len(val_s)}")

    fixed = {
        "K0_us_per_kernel": float(v4_params["K0_us_per_kernel"]),
        "H_prefill_us":     float(v4_params["H_prefill_us"]),
        "H_decode_us":      float(v4_params["H_decode_us"]),
        "aic_amp_decode":   0.85,
        "nk_mult_decode":   4.7,
        # AIV invariants (kept from v4 defaults)
        "aiv_amp_floor":    0.5,
        "aiv_amp_decode":   1.5,
        "aiv_C_kernel_us":  16.0,
        "aiv_C_data_us":    3.0,
        "aiv_active_frac":  0.85,
    }

    bounds = [physics_v5.V5_PARAM_BOUNDS[k] for k in _FIT_KEYS]
    defaults = [physics_v5.V5_PARAM_DEFAULTS[k] for k in _FIT_KEYS]

    def objective(theta):
        params = _make_params(theta.tolist(), fixed)
        return _mae(train, arch, params)

    print("\nFitting v5 on TRAIN (differential evolution)...")
    res = scipy.optimize.differential_evolution(
        objective, bounds=bounds, x0=defaults, seed=42,
        maxiter=80, tol=1e-3, popsize=15, polish=True, workers=1)
    best_theta = res.x.tolist()
    best_params = _make_params(best_theta, fixed)

    # Eval all splits
    print(f"\nBest TRAIN MAE = {res.fun:.2f}%  (after {res.nfev} evals)")
    print("\n=== TRAIN per-config errors ===")
    train_errs = []
    for cfg_name, spec, batch, meas in train:
        r = _eval_one(spec, arch, batch, best_params, meas)
        train_errs.append(r["wall_err_pct"])
        print(f"  {cfg_name:40s} wall_err={r['wall_err_pct']:>6.1f}%  "
              f"(aic={r['aic_err_pct']:.0f}% aiv={r['aiv_err_pct']:.0f}% nk={r['nk_err_pct']:.0f}%)")

    print("\n=== VAL_batch per-config errors ===")
    val_b_errs = []
    for cfg_name, spec, batch, meas in val_b:
        r = _eval_one(spec, arch, batch, best_params, meas)
        val_b_errs.append(r["wall_err_pct"])
        print(f"  {cfg_name:40s} wall_err={r['wall_err_pct']:>6.1f}%")

    print("\n=== VAL_size per-config errors ===")
    val_s_errs = []
    for cfg_name, spec, batch, meas in val_s:
        r = _eval_one(spec, arch, batch, best_params, meas)
        val_s_errs.append(r["wall_err_pct"])
        print(f"  {cfg_name:40s} wall_err={r['wall_err_pct']:>6.1f}%")

    summary = {
        "fitted_v5_params": {k: round(v, 4) for k, v in zip(_FIT_KEYS, best_theta)},
        "train_mae_pct": round(statistics.mean(train_errs), 2),
        "train_max_pct": round(max(train_errs), 2),
        "val_batch_mae_pct": round(statistics.mean(val_b_errs), 2) if val_b_errs else None,
        "val_batch_max_pct": round(max(val_b_errs), 2) if val_b_errs else None,
        "val_size_mae_pct": round(statistics.mean(val_s_errs), 2) if val_s_errs else None,
        "val_size_max_pct": round(max(val_s_errs), 2) if val_s_errs else None,
        "n_train": len(train),
        "n_val_batch": len(val_b),
        "n_val_size": len(val_s),
    }
    print(f"\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    return {
        "K0_us_per_kernel": fixed["K0_us_per_kernel"],
        "H_prefill_us":     fixed["H_prefill_us"],
        "H_decode_us":      fixed["H_decode_us"],
        "v_model":          "v5",
        **best_params,
        "fit_summary":      summary,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="data/calibration/pipe_baseline_per_model.json")
    p.add_argument("--arch", default="arch/ascend_910b4_for_sweep_v2.yaml")
    p.add_argument("--v4-params", default="data/calibration/predict_pipe_params.json")
    p.add_argument("--output", default="data/calibration/predict_pipe_params_v5.json")
    args = p.parse_args()

    result = fit_v5(args.baseline, args.arch, args.v4_params)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
