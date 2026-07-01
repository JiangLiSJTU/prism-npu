"""v7 LOMO (leave-one-model-out) cross-validation.

Parallel to lomo_v6.py but operates on v7's 3-bucket SDPA-aware calibration:
  - Iterates over all measured configs in TRAIN ∪ VAL_size ∪ VAL_sdpa_*
  - For each: remove it, refit ONLY its bucket on remaining configs in
    that bucket, predict held-out, record wall_clock err
  - For buckets that have only 1 anchor in TRAIN, removing it leaves 0
    train data → use V7_BUCKET_DEFAULTS (degraded baseline)

Closes Phase 2 step 8 (was missing per audit 2026-05-18).

Usage:
    python -m prism.predict_pipe.lomo_v7
"""
from __future__ import annotations
import dataclasses
import json
import statistics
from pathlib import Path

import scipy.optimize

from .model_spec import KNOWN_MODELS, ModelSpec
from .physics_v7 import (V7_BUCKET_BOUNDS, V7_BUCKET_DEFAULTS,
                         classify_bottleneck_v7)
from .predict import _arch_dict_from_yaml, predict_pipe_baseline
from .splits_v7 import (TRAIN_CONFIGS_V7, VAL_SIZE_V7,
                        VAL_SDPA_LONG_S_V7, VAL_SDPA_BATCH_V7)

_PARAMS_PER_BUCKET = ("amp_aic", "amp_aiv", "nk_mult")
_REPO = Path(__file__).resolve().parents[3]


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


def _build_full_dataset(baseline_doc):
    """All measured configs across TRAIN + all VAL_*: each kept once."""
    seen = set()
    ds = []
    all_configs = (list(TRAIN_CONFIGS_V7) + list(VAL_SIZE_V7)
                   + list(VAL_SDPA_LONG_S_V7) + list(VAL_SDPA_BATCH_V7))
    for cfg, yaml, batch in all_configs:
        if cfg in seen:
            continue
        seen.add(cfg)
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


def _fit_bucket(bucket, configs_in_bucket, arch, fixed):
    if not configs_in_bucket:
        return [V7_BUCKET_DEFAULTS[bucket][k] for k in _PARAMS_PER_BUCKET]

    bounds = [V7_BUCKET_BOUNDS[bucket][k] for k in _PARAMS_PER_BUCKET]
    x0 = [V7_BUCKET_DEFAULTS[bucket][k] for k in _PARAMS_PER_BUCKET]

    def obj(theta):
        p = dict(fixed)
        p["v_model"] = "v7"
        for key, val in zip(_PARAMS_PER_BUCKET, theta):
            p[f"v7_{bucket}_{key}"] = val
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
    print(f"v7 LOMO over {len(full)} measured configs")

    all_buckets = ("AIC_DECODE", "AIV_BOUND", "BALANCED")
    results = []
    for i, (cfg_name, _spec, _batch, _meas, _bucket) in enumerate(full):
        held = full[i]
        remaining = full[:i] + full[i+1:]

        per_bucket_theta = {}
        for b in all_buckets:
            in_bucket = [c for c in remaining if c[4] == b]
            per_bucket_theta[b] = _fit_bucket(b, in_bucket, arch, fixed)

        params = dict(fixed)
        params["v_model"] = "v7"
        for b in all_buckets:
            for key, val in zip(_PARAMS_PER_BUCKET, per_bucket_theta[b]):
                params[f"v7_{b}_{key}"] = val

        spec, batch, meas, bucket = held[1], held[2], held[3], held[4]
        pred = predict_pipe_baseline(spec, arch, params, batch=batch)
        err = abs(pred["wall_clock_us"] - meas["wall_clock_us"]) / meas["wall_clock_us"] * 100
        n_rem = len([c for c in remaining if c[4] == bucket])
        results.append({
            "config":                cfg_name,
            "bucket":                bucket,
            "n_remaining_in_bucket": n_rem,
            "wall_pred_us":          pred["wall_clock_us"],
            "wall_meas_us":          meas["wall_clock_us"],
            "wall_err_pct":          round(err, 2),
        })
        print(f"  [{i+1:2d}/{len(full)}] {cfg_name:42s} bucket={bucket:>10s} "
              f"n_rem={n_rem}  err={err:>6.1f}%")

    by_bucket = {}
    for r in results:
        by_bucket.setdefault(r["bucket"], []).append(r["wall_err_pct"])

    print("\n=== Per-bucket LOMO MAE ===")
    for b, errs in sorted(by_bucket.items()):
        print(f"  {b:>10s}: n={len(errs)} MAE={statistics.mean(errs):>6.1f}% max={max(errs):>6.1f}%")

    overall = statistics.mean([r["wall_err_pct"] for r in results])
    print(f"\n=== Overall v7 LOMO MAE = {overall:.2f}% ===")

    return {
        "results":              results,
        "overall_mae_pct":      round(overall, 2),
        "per_bucket_mae_pct":   {b: round(statistics.mean(e), 2) for b, e in by_bucket.items()},
        "n_configs":            len(results),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="data/calibration/pipe_baseline_per_model.json")
    p.add_argument("--arch",     default="arch/ascend_910b4_for_sweep_v2.yaml")
    p.add_argument("--v4-params",default="data/calibration/predict_pipe_params.json")
    p.add_argument("--output",   default="data/calibration/predict_pipe_v7_lomo.json")
    args = p.parse_args()
    result = run_lomo(args.baseline, args.arch, args.v4_params)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
