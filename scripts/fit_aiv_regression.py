#!/usr/bin/env python3
"""
Fit a msprof-driven multi-feature regression for AIV pipe times.

Replaces the v0.1 prototype's empirical ``aiv_time = 1.25 × aic_time`` heuristic
with per-pipe linear models in physical features:

    aiv_vec_us  ≈ A_vec  · vec_ops_phys     + B_vec  · n_kernels
    aiv_mte2_us ≈ A_mte2 · inter_bytes_phys + B_mte2 · n_kernels + C_mte2 · aic_mte2_us
    aiv_mte3_us ≈ A_mte3 · output_bytes_phys + B_mte3 · n_kernels

The aic_mte2 term in aiv_mte2 captures HBM contention — when the AIC side is
heavily loaded reading weights from HBM, AIV side accesses see lower effective
bandwidth. Confirmed by net_transformer-style msprof observations.

Outputs::

    data/calibration/predict_pipe_aiv_regression.json   # fitted coefficients
    stdout: per-pipe coefficients + LOO CV MAE summary
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from prism.predict_pipe.model_spec import KNOWN_MODELS, ModelSpec, compute_gemm_ops, compute_vector_ops


# ─────────────────────────────────────────────────────────────────────────
# 1. Wire all measured configs to ModelSpec (extending KNOWN_MODELS).
#    The same spec applies across b=1/4/8/16; batch is parsed from config name.
# ─────────────────────────────────────────────────────────────────────────
def _spec_alias(src_key: str, *new_keys: str) -> None:
    """Add aliases for existing KNOWN_MODELS spec (same spec, different batch)."""
    base = KNOWN_MODELS[src_key]
    for k in new_keys:
        if k not in KNOWN_MODELS:
            KNOWN_MODELS[k] = ModelSpec(**{**base.__dict__, "name": base.name})

_spec_alias("BERT-base-S128-b1",
            "BERT-base-S128-b4", "BERT-base-S128-b8", "BERT-base-S128-b16")
_spec_alias("GPT-2-S512-b1",
            "GPT-2-S512-b4", "GPT-2-S512-b8", "GPT-2-S512-b16")
_spec_alias("Qwen3-prefill-S256-b1",
            "Qwen3-prefill-S256-b4", "Qwen3-prefill-S256-b8")
_spec_alias("Qwen3-prefill-S512-b4",
            "Qwen3-prefill-S512-b8")

# Qwen3-prefill-S4096-b1 — distinct S, need new spec.
KNOWN_MODELS["Qwen3-prefill-S4096-b1"] = ModelSpec(
    name="Qwen3-0.6B", arch="decoder", layers=28, S=4096,
    d_model=1024, d_ff=3072, n_heads=16, n_kv_heads=8, d_head=128,
    vocab=151936, ffn_type="swiglu",
)

# HF BERT — 4-layer d=256 distilled encoder, fresh msprof from Wave 2.
_HF_BERT_BASE = ModelSpec(
    name="HF-BERT-distilled", arch="encoder", layers=4, S=128,
    d_model=256, d_ff=1024, n_heads=4, n_kv_heads=0, d_head=64,
    vocab=30522, ffn_type="standard",
)
for b in (1, 4, 8, 16):
    KNOWN_MODELS[f"HF-BERT-S128-b{b}"] = _HF_BERT_BASE

# Net-Transformer — 1-layer d=384 custom encoder (固定网络 baseline), Wave 3 msprof.
# ONNX input: [batch, 256, 384] FP32. Output: [batch, 1024] classifier (no large vocab head).
_NET_TRANS_BASE = ModelSpec(
    name="Net-Transformer", arch="encoder", layers=1, S=256,
    d_model=384, d_ff=1536, n_heads=6, n_kv_heads=0, d_head=64,
    vocab=1024,                          # 1024-class classifier head, NOT 10000-vocab embed
    ffn_type="standard",
)
_spec_alias_already_handled = False
for b in (1, 4, 8, 16):
    KNOWN_MODELS[f"Net-Transformer-S256-L1-b{b}"] = _NET_TRANS_BASE


def _batch_from_name(name: str) -> int:
    m = re.search(r"-b(\d+)$", name)
    return int(m.group(1)) if m else 1


# ─────────────────────────────────────────────────────────────────────────
# 2. Feature extraction.
# ─────────────────────────────────────────────────────────────────────────
def features_for_config(name: str, cfg: Dict) -> Dict[str, float]:
    """Build a feature row for one config: physics predictions + msprof aic side."""
    spec = KNOWN_MODELS[name]
    B = _batch_from_name(name)
    total_ops, act_read, weight_b, output_b = compute_gemm_ops(spec)
    vec_ops, inter_b = compute_vector_ops(spec)
    n_kernels = cfg["n_kernels_per_inf"]

    return {
        # Physics features (scaled by batch)
        "vec_ops_phys":         vec_ops * B,
        "intermediate_bytes":   inter_b * B,
        "output_bytes":         output_b * B,
        "n_kernels":            float(n_kernels),
        # AIC measurements (captures HBM contention, AIC workload size)
        "aic_mte2_us":          cfg["aic_pipes_us"]["mte2"],
        "aic_time_us":          cfg["aic_time_us"],
        # Derived for analysis
        "bytes_per_kernel":     inter_b * B / max(n_kernels, 1),
        "batch":                float(B),
        "S":                    float(spec.S),
    }


def collect_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Return (config_names, features dict per config, targets dict per config)."""
    baseline_path = _REPO / "data" / "calibration" / "pipe_baseline_per_model.json"
    with open(baseline_path, encoding="utf-8") as f:
        doc = json.load(f)
    configs = doc["configs"]

    names, feats, tgts = [], {}, {}
    for name, cfg in sorted(configs.items()):
        if cfg.get("source", "").startswith(("estimated", "inherited")):
            continue
        if name not in KNOWN_MODELS:
            print(f"  skip (no ModelSpec): {name}", file=sys.stderr)
            continue
        names.append(name)
        feats[name] = features_for_config(name, cfg)
        tgts[name] = {
            "vec":     cfg["aiv_pipes_us"]["vec"],
            "mte2":    cfg["aiv_pipes_us"]["mte2"],
            "mte3":    cfg["aiv_pipes_us"]["mte3"],
            "scalar":  cfg["aiv_pipes_us"]["scalar"],
            "idle":    cfg["aiv_pipes_us"].get("idle", 0.0),
            "aiv_time": cfg["aiv_time_us"],
        }
    return names, feats, tgts


# ─────────────────────────────────────────────────────────────────────────
# 3. Non-negative least squares (forces physical coefficients ≥ 0).
# ─────────────────────────────────────────────────────────────────────────
def _fit_nnls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve min ||Xβ - y||² s.t. β ≥ 0 via scipy.optimize.nnls."""
    from scipy.optimize import nnls
    coef, _ = nnls(X, y)
    return coef


def _design_matrix(feats: Dict[str, Dict[str, float]],
                   names: List[str],
                   feature_keys: List[str]) -> np.ndarray:
    return np.array([[feats[n][k] for k in feature_keys] for n in names])


def _mae_pct(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    nonzero = y_true > 1e-6
    return float(np.mean(np.abs(y_pred[nonzero] - y_true[nonzero]) / y_true[nonzero]) * 100)


# ─────────────────────────────────────────────────────────────────────────
# 4. Per-pipe regression + LOO CV.
# ─────────────────────────────────────────────────────────────────────────
PIPE_FEATURES = {
    "vec":  ["vec_ops_phys", "n_kernels"],
    "mte2": ["intermediate_bytes", "n_kernels", "aic_mte2_us"],
    "mte3": ["output_bytes", "n_kernels"],
}


def fit_pipe(pipe: str, names: List[str],
             feats: Dict[str, Dict[str, float]],
             tgts: Dict[str, Dict[str, float]]) -> Dict:
    feature_keys = PIPE_FEATURES[pipe]
    X = _design_matrix(feats, names, feature_keys)
    y = np.array([tgts[n][pipe] for n in names])
    coef = _fit_nnls(X, y)
    y_pred = X @ coef
    train_mae = _mae_pct(y_pred, y)

    # LOO CV
    loo_errors = []
    for i in range(len(names)):
        idx_tr = [j for j in range(len(names)) if j != i]
        coef_loo = _fit_nnls(X[idx_tr], y[idx_tr])
        y_test_pred = X[i] @ coef_loo
        if y[i] > 1e-6:
            loo_errors.append(abs(y_test_pred - y[i]) / y[i] * 100)
    loo_mae = float(np.mean(loo_errors)) if loo_errors else 0.0

    return {
        "feature_keys": feature_keys,
        "coefficients": [float(c) for c in coef],
        "train_mae_pct": train_mae,
        "loo_cv_mae_pct": loo_mae,
        "per_config_pred": {n: float(y_pred[i]) for i, n in enumerate(names)},
        "per_config_actual": {n: float(y[i]) for i, n in enumerate(names)},
    }


def fit_idle(names: List[str],
             feats: Dict[str, Dict[str, float]],
             tgts: Dict[str, Dict[str, float]]) -> Dict:
    """idle ≈ A · n_kernels + B · (aic_time - aiv_active_pipes).
    Captures stall time when AIV waits on data."""
    feature_keys = ["n_kernels"]
    X = _design_matrix(feats, names, feature_keys)
    y = np.array([tgts[n]["idle"] for n in names])
    coef = _fit_nnls(X, y)
    y_pred = X @ coef
    return {
        "feature_keys": feature_keys,
        "coefficients": [float(c) for c in coef],
        "train_mae_pct": _mae_pct(y_pred, y),
    }


def fit_scalar(names: List[str],
               feats: Dict[str, Dict[str, float]],
               tgts: Dict[str, Dict[str, float]]) -> Dict:
    """scalar ≈ A · n_kernels (mostly per-kernel cost)."""
    feature_keys = ["n_kernels"]
    X = _design_matrix(feats, names, feature_keys)
    y = np.array([tgts[n]["scalar"] for n in names])
    coef = _fit_nnls(X, y)
    y_pred = X @ coef
    return {
        "feature_keys": feature_keys,
        "coefficients": [float(c) for c in coef],
        "train_mae_pct": _mae_pct(y_pred, y),
    }


def compare_with_baseline(names: List[str], tgts: Dict[str, Dict[str, float]],
                          feats: Dict[str, Dict[str, float]]) -> Dict:
    """Compute baseline (1.25 × aic_time) error and the new model's aiv_time error."""
    pipe_fits = {p: fit_pipe(p, names, feats, tgts) for p in PIPE_FEATURES}
    idle_fit = fit_idle(names, feats, tgts)
    scalar_fit = fit_scalar(names, feats, tgts)

    # Reconstruct predicted aiv_time = max(vec, mte2, mte3, scalar) + idle
    # using regression. Compare to 1.25 × aic_time baseline.
    new_errs, old_errs = [], []
    rows = []
    for n in names:
        actual = tgts[n]["aiv_time"]
        old_pred = 1.25 * feats[n]["aic_time_us"]
        # New: sum of fitted pipes (since AIV pipes are sequential dispatch in time)
        v = pipe_fits["vec"]["per_config_pred"][n]
        m2 = pipe_fits["mte2"]["per_config_pred"][n]
        m3 = pipe_fits["mte3"]["per_config_pred"][n]
        sc = scalar_fit["coefficients"][0] * feats[n]["n_kernels"]
        idle = idle_fit["coefficients"][0] * feats[n]["n_kernels"]
        # The pipes don't all execute concurrently; observed aiv_time ≈ max(pipes) + idle
        # is a coarse but useful aggregate. Use sum since pipes serialize per-kernel.
        new_pred = max(v, m2, m3, sc) + idle
        old_err = abs(old_pred - actual) / actual * 100 if actual else 0
        new_err = abs(new_pred - actual) / actual * 100 if actual else 0
        old_errs.append(old_err); new_errs.append(new_err)
        rows.append((n, actual, old_pred, old_err, new_pred, new_err))

    return {
        "pipe_fits": pipe_fits,
        "idle_fit": idle_fit,
        "scalar_fit": scalar_fit,
        "baseline_125x_aic": {"mae_pct": float(np.mean(old_errs))},
        "new_regression":    {"mae_pct": float(np.mean(new_errs))},
        "rows": rows,
    }


def main() -> int:
    names, feats, tgts = collect_data()
    print(f"Loaded {len(names)} measured configs:")
    for n in names:
        print(f"  {n}")

    out = compare_with_baseline(names, tgts, feats)

    print()
    print("=" * 100)
    print("PER-PIPE REGRESSION FITS (non-negative least squares)")
    print("=" * 100)
    for pipe, fit in out["pipe_fits"].items():
        print(f"\nAIV-{pipe}:  features = {fit['feature_keys']}")
        for k, c in zip(fit["feature_keys"], fit["coefficients"]):
            print(f"           coef[{k}] = {c:.4e}")
        print(f"           train MAE = {fit['train_mae_pct']:.1f}%   LOO CV MAE = {fit['loo_cv_mae_pct']:.1f}%")

    print(f"\nAIV-idle:   coef[n_kernels] = {out['idle_fit']['coefficients'][0]:.4f} μs/kernel, train MAE {out['idle_fit']['train_mae_pct']:.1f}%")
    print(f"AIV-scalar: coef[n_kernels] = {out['scalar_fit']['coefficients'][0]:.4f} μs/kernel, train MAE {out['scalar_fit']['train_mae_pct']:.1f}%")

    print()
    print("=" * 100)
    print(f"BASELINE (1.25×aic_time): aiv_time MAE = {out['baseline_125x_aic']['mae_pct']:.1f}%")
    print(f"NEW REGRESSION model:      aiv_time MAE = {out['new_regression']['mae_pct']:.1f}%")
    print("=" * 100)
    print(f"\n{'Config':<35} {'meas':>10} {'old_pred':>10} {'old_err%':>10} {'new_pred':>10} {'new_err%':>10}")
    for n, actual, old_pred, old_err, new_pred, new_err in out["rows"]:
        print(f"{n:<35} {actual:>10.0f} {old_pred:>10.0f} {old_err:>9.1f}% {new_pred:>10.0f} {new_err:>9.1f}%")

    # Persist coefficients (without per_config arrays)
    persist = {
        "fit_version": "v2_msprof_regression",
        "n_configs": len(names),
        "configs": names,
        "pipes": {p: {"features": f["feature_keys"], "coefficients": f["coefficients"],
                      "train_mae_pct": f["train_mae_pct"], "loo_cv_mae_pct": f["loo_cv_mae_pct"]}
                  for p, f in out["pipe_fits"].items()},
        "idle":   {"features": out["idle_fit"]["feature_keys"], "coefficients": out["idle_fit"]["coefficients"]},
        "scalar": {"features": out["scalar_fit"]["feature_keys"], "coefficients": out["scalar_fit"]["coefficients"]},
        "baseline_125x_aic_mae_pct": out["baseline_125x_aic"]["mae_pct"],
        "new_regression_mae_pct": out["new_regression"]["mae_pct"],
    }
    out_path = _REPO / "data" / "calibration" / "predict_pipe_aiv_regression.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(persist, f, ensure_ascii=False, indent=2)
    print(f"\nWritten to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
