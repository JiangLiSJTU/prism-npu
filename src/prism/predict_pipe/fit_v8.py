"""v8 fit — multi-objective per-bucket calibration (user mandate 2026-05-18).

User: "我希望泛化能力强，同时各个部件的仿真误差尽量小"

v6 / v7 issue (per audit docs/findings/predict_pipe_component_cancellation_audit.md):
fit objective is wall_clock MAE only → DE optimizer freely chooses (amp_aic,
amp_aiv) pair that cancel each other. Result: v6 TRAIN AIC 47.7% + AIV 45.7%
canceled to wall 0.2% — cancellation_ratio = 204.

v8 fix: composite loss penalizes component errors directly:

    loss = wall_mae + λ_aic·aic_mae + λ_aiv·aiv_mae + λ_nk·nk_mae

Default λ = (0.3, 0.3, 0.2). Pushes optimizer toward solutions where each
component tracks its own measurement, not just the sum.

Builds on v7 (SDPA baseline, 3-bucket structure) since SDPA is the
production-accurate measurement path.

Output: data/calibration/predict_pipe_params_v8.json
"""
from __future__ import annotations
import dataclasses
import json
import statistics
from pathlib import Path
from typing import Dict

import scipy.optimize

from . import physics_v7
from .model_spec import ModelSpec, KNOWN_MODELS
from .physics_v7 import (V7_BUCKET_BOUNDS, V7_BUCKET_DEFAULTS,
                         classify_bottleneck_v7)
from .predict import _arch_dict_from_yaml, predict_pipe_baseline
from .splits_v7 import (TRAIN_CONFIGS_V7, VAL_SIZE_V7,
                        VAL_SDPA_LONG_S_V7, VAL_SDPA_BATCH_V7)

_PARAMS_PER_BUCKET = ("amp_aic", "amp_aiv", "nk_mult")
_REPO = Path(__file__).resolve().parents[3]

# Multi-objective weights (user mandate 2026-05-18)
# Total weight = 1.0 + sum(λ) = 1.0 + 0.3 + 0.3 + 0.2 = 1.8
# Wall remains primary signal but components contribute substantively.
LAMBDA_AIC = 0.3
LAMBDA_AIV = 0.3
LAMBDA_NK = 0.2


def _qwen3_with_S(S: int):
    base = KNOWN_MODELS["Qwen3-prefill-S256-b1"]
    return dataclasses.replace(base, S=S, name=f"Qwen3-prefill-S{S}")


def _load_spec(cfg_name, yaml_path):
    base_name_match = cfg_name.replace("-sdpa", "")
    if base_name_match in KNOWN_MODELS:
        return KNOWN_MODELS[base_name_match]
    base = base_name_match.rsplit("-b", 1)[0] + "-b1"
    if base in KNOWN_MODELS:
        return KNOWN_MODELS[base]
    if base_name_match.startswith("Qwen3-prefill-S"):
        try:
            return _qwen3_with_S(int(base_name_match.split("-S")[1].split("-")[0]))
        except (IndexError, ValueError):
            pass
    try:
        return ModelSpec.from_yaml(_REPO / yaml_path)
    except (KeyError, FileNotFoundError):
        return None


def _build_dataset(configs, baseline_doc):
    ds = []
    for cfg, yaml, batch in configs:
        if cfg not in baseline_doc["configs"]:
            continue
        meas = baseline_doc["configs"][cfg]
        if not meas.get("wall_clock_us"):
            continue
        spec = _load_spec(cfg, yaml)
        if spec is None:
            continue
        bucket = classify_bottleneck_v7(spec, batch)
        ds.append((cfg, spec, batch, meas, bucket))
    return ds


def _make_params(per_bucket_theta, fixed):
    p = dict(fixed)
    p["v_model"] = "v8"  # ← also enable v7 dispatch (v8 reuses v7 schema)
    for bucket, theta in per_bucket_theta.items():
        for key, val in zip(_PARAMS_PER_BUCKET, theta):
            p[f"v7_{bucket}_{key}"] = val  # KEEP v7_ prefix — predict.py dispatches on this
    return p


def _per_config_err(spec, batch, meas, params, attn_impl="eager"):
    """Return (wall_err, aic_err, aiv_err, nk_err) all in percent.

    Issue #11: attn_impl threads through to predict_pipe_baseline so SDPA
    configs use the per-family multivariate host_gap fit.
    """
    pred = predict_pipe_baseline(spec, arch_dict, params, batch=batch, attn_impl=attn_impl)

    def _e(pred_v, meas_v):
        return abs(pred_v - meas_v) / max(meas_v, 1) * 100

    return (
        _e(pred["wall_clock_us"],       meas["wall_clock_us"]),
        _e(pred["aic_time_us"],         meas["aic_time_us"]),
        _e(pred["aiv_time_us"],         meas["aiv_time_us"]),
        _e(pred["n_kernels_per_inf"],   meas["n_kernels_per_inf"]),
    )


# Will be set in fit_v8()
arch_dict = None


def _multi_objective_loss(theta, bucket, configs_in_bucket,
                          other_buckets_params, fixed):
    """Composite loss = wall + λ·(aic + aiv + nk)."""
    per_bucket = dict(other_buckets_params)
    per_bucket[bucket] = theta
    params = _make_params(per_bucket, fixed)

    wall_errs, aic_errs, aiv_errs, nk_errs = [], [], [], []
    for cfg, spec, batch, meas, _b in configs_in_bucket:
        w, a, v, n = _per_config_err(spec, batch, meas, params)
        wall_errs.append(w)
        aic_errs.append(a)
        aiv_errs.append(v)
        nk_errs.append(n)

    return (
        statistics.mean(wall_errs)
        + LAMBDA_AIC * statistics.mean(aic_errs)
        + LAMBDA_AIV * statistics.mean(aiv_errs)
        + LAMBDA_NK  * statistics.mean(nk_errs)
    )


def _wall_err_pct(pred_us, meas_us):
    return abs(pred_us - meas_us) / max(meas_us, 1) * 100


def fit_v8(baseline_path, arch_yaml, v4_params_path) -> Dict:
    global arch_dict
    baseline = json.load(open(baseline_path, encoding="utf-8"))
    arch_dict = _arch_dict_from_yaml(arch_yaml)
    v4 = json.load(open(v4_params_path, encoding="utf-8"))

    train = _build_dataset(TRAIN_CONFIGS_V7, baseline)
    val_size = _build_dataset(VAL_SIZE_V7, baseline)
    val_long_s = _build_dataset(VAL_SDPA_LONG_S_V7, baseline)
    val_batch = _build_dataset(VAL_SDPA_BATCH_V7, baseline)
    print(f"Datasets: TRAIN={len(train)}  VAL_size={len(val_size)}  "
          f"VAL_sdpa_long_S={len(val_long_s)}  VAL_sdpa_batch={len(val_batch)}")
    print(f"Multi-objective weights: λ_aic={LAMBDA_AIC}  λ_aiv={LAMBDA_AIV}  λ_nk={LAMBDA_NK}")

    buckets = sorted(set(b for _,_,_,_,b in train))
    by_bucket = {b: [c for c in train if c[4] == b] for b in buckets}
    for b, cs in by_bucket.items():
        print(f"  {b}: {len(cs)} configs → {[c[0] for c in cs]}")

    fixed = {
        "K0_us_per_kernel": float(v4["K0_us_per_kernel"]),
        "H_prefill_us":     float(v4["H_prefill_us"]),
        "H_decode_us":      float(v4["H_decode_us"]),
        "aiv_C_kernel_us":  16.0,
        "aiv_C_data_us":    3.0,
    }

    per_bucket_theta = {
        b: [V7_BUCKET_DEFAULTS[b][k] for k in _PARAMS_PER_BUCKET]
        for b in V7_BUCKET_DEFAULTS.keys()
    }

    print("\n=== Per-bucket multi-objective DE fit ===")
    for bucket in buckets:
        if not by_bucket[bucket]:
            continue
        bounds = [V7_BUCKET_BOUNDS[bucket][k] for k in _PARAMS_PER_BUCKET]
        x0 = per_bucket_theta[bucket]

        def obj(theta):
            return _multi_objective_loss(theta.tolist(), bucket,
                                          by_bucket[bucket], per_bucket_theta, fixed)

        res = scipy.optimize.differential_evolution(
            obj, bounds=bounds, x0=x0, seed=42, maxiter=80, tol=1e-3,
            popsize=15, polish=True, workers=1)
        per_bucket_theta[bucket] = res.x.tolist()
        print(f"  {bucket}: composite_loss={res.fun:.2f}  theta={[round(x,3) for x in res.x.tolist()]}")

    best_params = _make_params(per_bucket_theta, fixed)

    def eval_set(name, ds):
        wall_e, aic_e, aiv_e, nk_e = [], [], [], []
        per_config = []
        for cfg, spec, batch, meas, bucket in ds:
            w, a, v, n = _per_config_err(spec, batch, meas, best_params)
            wall_e.append(w); aic_e.append(a); aiv_e.append(v); nk_e.append(n)
            per_config.append((cfg, bucket, w, a, v, n))
        return wall_e, aic_e, aiv_e, nk_e, per_config

    def _report(name, pc):
        print(f"\n=== {name} ===")
        for cfg, bucket, w, a, v, n in pc:
            print(f"  {cfg:42s} [{bucket:>10s}] wall={w:>6.1f}% aic={a:>6.1f}% aiv={v:>6.1f}% nk={n:>6.1f}%")

    train_w, train_a, train_v, train_n, train_pc = eval_set("TRAIN", train)
    vs_w, vs_a, vs_v, vs_n, vs_pc = eval_set("VAL_size", val_size)
    vl_w, vl_a, vl_v, vl_n, vl_pc = eval_set("VAL_sdpa_long_S", val_long_s)
    vb_w, vb_a, vb_v, vb_n, vb_pc = eval_set("VAL_sdpa_batch", val_batch)

    _report("TRAIN", train_pc)
    _report("VAL_size", vs_pc)
    _report("VAL_sdpa_long_S", vl_pc)
    _report("VAL_sdpa_batch", vb_pc)

    def _stats(errs):
        return {
            "mae_pct": round(statistics.mean(errs), 2) if errs else None,
            "max_pct": round(max(errs), 2) if errs else None,
        }

    summary = {
        "objective":            "multi-objective: wall + 0.3·aic + 0.3·aiv + 0.2·nk",
        "lambda_aic":           LAMBDA_AIC,
        "lambda_aiv":           LAMBDA_AIV,
        "lambda_nk":            LAMBDA_NK,
        "per_bucket_fit": {
            b: {k: round(v, 4) for k, v in zip(_PARAMS_PER_BUCKET, per_bucket_theta[b])}
            for b in V7_BUCKET_DEFAULTS.keys()
        },
        "train":            {"wall": _stats(train_w), "aic": _stats(train_a),
                             "aiv": _stats(train_v), "nk": _stats(train_n)},
        "val_size":         {"wall": _stats(vs_w),    "aic": _stats(vs_a),
                             "aiv": _stats(vs_v),    "nk": _stats(vs_n)},
        "val_sdpa_long_s":  {"wall": _stats(vl_w),    "aic": _stats(vl_a),
                             "aiv": _stats(vl_v),    "nk": _stats(vl_n)},
        "val_sdpa_batch":   {"wall": _stats(vb_w),    "aic": _stats(vb_a),
                             "aiv": _stats(vb_v),    "nk": _stats(vb_n)},
        "n_train": len(train), "n_val_size": len(val_size),
        "n_val_sdpa_long_s": len(val_long_s), "n_val_sdpa_batch": len(val_batch),
    }
    print(f"\n=== v8 Summary ===")
    print(json.dumps(summary, indent=2))

    return {**best_params, "fit_summary": summary}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="data/calibration/pipe_baseline_per_model.json")
    p.add_argument("--arch",     default="arch/ascend_910b4_for_sweep_v2.yaml")
    p.add_argument("--v4-params",default="data/calibration/predict_pipe_params.json")
    p.add_argument("--output",   default="data/calibration/predict_pipe_params_v8.json")
    args = p.parse_args()
    result = fit_v8(args.baseline, args.arch, args.v4_params)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
