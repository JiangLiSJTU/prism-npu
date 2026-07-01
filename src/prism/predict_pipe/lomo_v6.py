"""v6 LOMO (leave-one-model-out) cross-validation.

For each measured config in TRAIN ∪ VAL_*:
  1. Mark it as held-out
  2. Refit ONLY its bucket (other buckets keep their full-fit values)
  3. Predict on held-out, record wall_clock err

This gives a per-config OOS err estimate that doesn't peek at the held-out
config during fitting. For buckets that have only 1 anchor in TRAIN, removing
that anchor leaves 0 train data → use V6_BUCKET_DEFAULTS (degraded baseline).

Usage:
    python -m prism.predict_pipe.lomo_v6
"""
from __future__ import annotations
import dataclasses
import json
import statistics
from pathlib import Path

import scipy.optimize

from .model_spec import KNOWN_MODELS, ModelSpec
from .physics_v6 import (V6_BUCKET_BOUNDS, V6_BUCKET_DEFAULTS,
                         classify_bottleneck)
from .predict import _arch_dict_from_yaml, predict_pipe_baseline
from .splits import (TRAIN_CONFIGS, VAL_BATCH_CONFIGS, VAL_SIZE_CONFIGS,
                     resolve_path)

_PARAMS_PER_BUCKET = ("amp_aic", "amp_aiv", "nk_mult", "amp_aic_S_alpha", "amp_aiv_S_alpha")


def _qwen3_with_S(S: int):
    base = KNOWN_MODELS["Qwen3-prefill-S256-b1"]
    return dataclasses.replace(base, S=S, name=f"Qwen3-prefill-S{S}")


def _load_spec(cfg_name, yaml_path):
    if cfg_name in KNOWN_MODELS:
        return KNOWN_MODELS[cfg_name]
    base = cfg_name.rsplit("-b", 1)[0] + "-b1"
    if base in KNOWN_MODELS:
        return KNOWN_MODELS[base]
    if cfg_name.startswith("Qwen3-prefill-S"):
        try:
            return _qwen3_with_S(int(cfg_name.split("-S")[1].split("-")[0]))
        except (IndexError, ValueError):
            pass
    try:
        return ModelSpec.from_yaml(resolve_path(yaml_path))
    except (KeyError, FileNotFoundError):
        return None


def _build_full_dataset(baseline_doc):
    ds = []
    for cfg, yaml, batch in TRAIN_CONFIGS + VAL_BATCH_CONFIGS + VAL_SIZE_CONFIGS:
        if cfg not in baseline_doc["configs"]:
            continue
        meas = baseline_doc["configs"][cfg]
        if not meas.get("wall_clock_us"):
            continue
        spec = _load_spec(cfg, yaml)
        if spec is None:
            continue
        bucket = classify_bottleneck(spec, batch)
        ds.append((cfg, spec, batch, meas, bucket))
    return ds


def _fit_bucket(bucket, configs_in_bucket, arch, fixed):
    """Fit one bucket on its configs. Returns theta list."""
    if not configs_in_bucket:
        # No data → use defaults
        return [V6_BUCKET_DEFAULTS[bucket][k] for k in _PARAMS_PER_BUCKET]

    bounds = [V6_BUCKET_BOUNDS[bucket][k] for k in _PARAMS_PER_BUCKET]
    x0 = [V6_BUCKET_DEFAULTS[bucket][k] for k in _PARAMS_PER_BUCKET]

    def obj(theta):
        # Build params with just this bucket's theta + global fixed
        p = dict(fixed)
        p["v_model"] = "v6"
        for key, val in zip(_PARAMS_PER_BUCKET, theta):
            p[f"v6_{bucket}_{key}"] = val
        errs = []
        for cfg, spec, batch, meas, _b in configs_in_bucket:
            pred = predict_pipe_baseline(spec, arch, p, batch=batch)
            mw = meas["wall_clock_us"]
            errs.append(abs(pred["wall_clock_us"] - mw) / mw * 100)
        return statistics.mean(errs)

    res = scipy.optimize.differential_evolution(
        obj, bounds=bounds, x0=x0, seed=42, maxiter=40, tol=1e-3,
        popsize=10, polish=True, workers=1)
    return res.x.tolist()


def run_lomo(baseline_path, arch_yaml, v4_params_path) -> dict:
    baseline = json.load(open(baseline_path))
    arch = _arch_dict_from_yaml(arch_yaml)
    v4 = json.load(open(v4_params_path))
    fixed = {
        "K0_us_per_kernel": float(v4["K0_us_per_kernel"]),
        "H_prefill_us":     float(v4["H_prefill_us"]),
        "H_decode_us":      float(v4["H_decode_us"]),
        "aiv_C_kernel_us":  16.0,
        "aiv_C_data_us":    3.0,
    }

    full = _build_full_dataset(baseline)
    print(f"LOMO over {len(full)} measured configs")

    results = []
    for i, (cfg_name, _spec, _batch, _meas, _bucket) in enumerate(full):
        held_out = full[i]
        remaining = full[:i] + full[i+1:]

        # Refit each bucket on its remaining configs only
        all_buckets = ("AIC_DECODE", "AIC_QWEN3", "AIV_BOUND", "BALANCED")
        per_bucket_theta = {}
        for b in all_buckets:
            in_bucket = [c for c in remaining if c[4] == b]
            per_bucket_theta[b] = _fit_bucket(b, in_bucket, arch, fixed)

        # Build params for prediction
        params = dict(fixed)
        params["v_model"] = "v6"
        for b in all_buckets:
            for key, val in zip(_PARAMS_PER_BUCKET, per_bucket_theta[b]):
                params[f"v6_{b}_{key}"] = val

        # Predict held-out
        spec = held_out[1]
        batch = held_out[2]
        meas = held_out[3]
        bucket = held_out[4]
        pred = predict_pipe_baseline(spec, arch, params, batch=batch)
        err = abs(pred["wall_clock_us"] - meas["wall_clock_us"]) / meas["wall_clock_us"] * 100
        train_size = len([c for c in remaining if c[4] == bucket])
        results.append({
            "config": cfg_name,
            "bucket": bucket,
            "n_remaining_in_bucket": train_size,
            "wall_pred_us": pred["wall_clock_us"],
            "wall_meas_us": meas["wall_clock_us"],
            "wall_err_pct": round(err, 2),
        })
        print(f"  [{i+1:2d}/{len(full)}] {cfg_name:40s} bucket={bucket:>10s} "
              f"n_remain={train_size}  err={err:>6.1f}%")

    # Per-bucket aggregate
    by_bucket = {}
    for r in results:
        by_bucket.setdefault(r["bucket"], []).append(r["wall_err_pct"])

    print("\n=== Per-bucket LOMO MAE ===")
    for b, errs in sorted(by_bucket.items()):
        if not errs:
            continue
        print(f"  {b:>10s}: n={len(errs)} MAE={statistics.mean(errs):>6.1f}% max={max(errs):>6.1f}%")

    overall_mae = statistics.mean([r["wall_err_pct"] for r in results])
    print(f"\n=== Overall LOMO MAE = {overall_mae:.2f}% ===")

    return {"results": results, "overall_mae_pct": round(overall_mae, 2),
            "per_bucket_mae_pct": {b: round(statistics.mean(errs), 2)
                                    for b, errs in by_bucket.items()}}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="data/calibration/pipe_baseline_per_model.json")
    p.add_argument("--arch", default="arch/ascend_910b4_for_sweep_v2.yaml")
    p.add_argument("--v4-params", default="data/calibration/predict_pipe_params.json")
    p.add_argument("--output", default="data/calibration/predict_pipe_v6_lomo.json")
    args = p.parse_args()
    result = run_lomo(args.baseline, args.arch, args.v4_params)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
