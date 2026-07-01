"""v6 fit — per-bucket coefficient calibration.

Responds to user 2026-05-17 insight: different bottleneck regimes need
separate calibration. v5 failed because single continuous formula can't
satisfy Qwen3-prefill (high aic_amp needed) + Llama (low aic_amp) at
similar w_proxy.

v6 strategy:
  1. Classify each config into one of 4 buckets (physics_v6.classify_bottleneck)
  2. Per-bucket fit (3 params each): amp_aic, amp_aiv, nk_mult
  3. Each bucket fit on its OWN train subset → no cross-bucket leakage

Train/val splits inherit from splits.py but get re-grouped by bucket.

Acceptance bar (vs v5):
  - TRAIN MAE ≤ 15% (v5: 17.3%)
  - VAL_size MAE ≤ 30% (v5: 104%)
  - Llama wall_err ≤ 50% (v5: 232%)
"""
from __future__ import annotations
import dataclasses
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import scipy.optimize

from . import physics_v6
from .model_spec import ModelSpec, KNOWN_MODELS
from .physics_v6 import (V6_BUCKET_BOUNDS, V6_BUCKET_DEFAULTS,
                         classify_bottleneck, get_bucket_params)
from .predict import _arch_dict_from_yaml, predict_pipe_baseline
from .splits import (TRAIN_CONFIGS, VAL_BATCH_CONFIGS, VAL_SIZE_CONFIGS,
                     resolve_path)

_PARAMS_PER_BUCKET = ("amp_aic", "amp_aiv", "nk_mult", "amp_aic_S_alpha", "amp_aiv_S_alpha")


def _qwen3_with_S(S: int) -> ModelSpec:
    base = KNOWN_MODELS["Qwen3-prefill-S256-b1"]
    return dataclasses.replace(base, S=S, name=f"Qwen3-prefill-S{S}")


def _load_spec(cfg_name: str, yaml_path: str):
    if cfg_name in KNOWN_MODELS:
        return KNOWN_MODELS[cfg_name]
    base_name = cfg_name.rsplit("-b", 1)[0] + "-b1"
    if base_name in KNOWN_MODELS:
        return KNOWN_MODELS[base_name]
    if cfg_name.startswith("Qwen3-prefill-S"):
        try:
            S_str = cfg_name.split("-S")[1].split("-")[0]
            return _qwen3_with_S(int(S_str))
        except (IndexError, ValueError):
            pass
    try:
        return ModelSpec.from_yaml(resolve_path(yaml_path))
    except (KeyError, FileNotFoundError):
        return None


def _build_dataset(configs, baseline_doc):
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
        bucket = classify_bottleneck(spec, batch)
        dataset.append((cfg_name, spec, batch, meas, bucket))
    return dataset


def _make_v6_params(per_bucket_theta: Dict[str, List[float]],
                    fixed: Dict[str, float]) -> Dict[str, float]:
    """Flatten per-bucket theta into v6_BUCKET_param keys."""
    p = dict(fixed)
    p["v_model"] = "v6"
    for bucket, theta in per_bucket_theta.items():
        for key, val in zip(_PARAMS_PER_BUCKET, theta):
            p[f"v6_{bucket}_{key}"] = val
    return p


def _wall_err_pct(pred_us: float, meas_us: float) -> float:
    if not meas_us:
        return 0.0
    return abs(pred_us - meas_us) / meas_us * 100


def _bucket_loss(theta: List[float], bucket: str, configs_in_bucket: list,
                 arch, other_buckets_params: Dict[str, List[float]],
                 fixed: Dict[str, float]) -> float:
    """MAE for one bucket given theta, holding other buckets fixed."""
    per_bucket = dict(other_buckets_params)
    per_bucket[bucket] = theta
    params = _make_v6_params(per_bucket, fixed)
    errs = []
    for cfg_name, spec, batch, meas, _b in configs_in_bucket:
        pred = predict_pipe_baseline(spec, arch, params, batch=batch)
        errs.append(_wall_err_pct(pred["wall_clock_us"], meas["wall_clock_us"]))
    return statistics.mean(errs) if errs else 0.0


def fit_v6(baseline_path, arch_yaml, v4_params_path) -> Dict:
    baseline = json.load(open(baseline_path, encoding="utf-8"))
    arch = _arch_dict_from_yaml(arch_yaml)
    v4_params = json.load(open(v4_params_path, encoding="utf-8"))

    train_all = _build_dataset(TRAIN_CONFIGS, baseline)
    val_b_all = _build_dataset(VAL_BATCH_CONFIGS, baseline)
    val_s_all = _build_dataset(VAL_SIZE_CONFIGS, baseline)
    print(f"Dataset sizes: TRAIN={len(train_all)} VAL_batch={len(val_b_all)} VAL_size={len(val_s_all)}")

    # Group TRAIN by bucket (for per-bucket fitting)
    buckets = sorted(set(b for _,_,_,_,b in train_all))
    print(f"Buckets in TRAIN: {buckets}")
    by_bucket = {b: [c for c in train_all if c[4] == b] for b in buckets}
    for b, cs in by_bucket.items():
        names = [c[0] for c in cs]
        print(f"  {b}: {len(cs)} configs → {names}")

    fixed = {
        "K0_us_per_kernel": float(v4_params["K0_us_per_kernel"]),
        "H_prefill_us":     float(v4_params["H_prefill_us"]),
        "H_decode_us":      float(v4_params["H_decode_us"]),
        "aiv_C_kernel_us":  16.0,
        "aiv_C_data_us":    3.0,
    }

    # Initialize all buckets to defaults
    per_bucket_theta = {
        b: [V6_BUCKET_DEFAULTS[b][k] for k in _PARAMS_PER_BUCKET]
        for b in V6_BUCKET_DEFAULTS.keys()
    }

    # Fit each bucket independently (held-others-fixed style)
    print("\n=== Per-bucket DE fit ===")
    for bucket in buckets:
        if not by_bucket[bucket]:
            print(f"  {bucket}: no train configs, keep defaults")
            continue
        bounds = [V6_BUCKET_BOUNDS[bucket][k] for k in _PARAMS_PER_BUCKET]
        x0 = per_bucket_theta[bucket]

        def obj(theta):
            return _bucket_loss(theta.tolist(), bucket, by_bucket[bucket],
                                arch, per_bucket_theta, fixed)

        res = scipy.optimize.differential_evolution(
            obj, bounds=bounds, x0=x0, seed=42, maxiter=60,
            tol=1e-3, popsize=15, polish=True, workers=1)
        per_bucket_theta[bucket] = res.x.tolist()
        print(f"  {bucket}: MAE={res.fun:.2f}%  theta={[round(x,3) for x in res.x.tolist()]}")

    # Final evaluation
    best_params = _make_v6_params(per_bucket_theta, fixed)

    def eval_set(name, ds):
        errs = []
        per_config = []
        for cfg_name, spec, batch, meas, bucket in ds:
            pred = predict_pipe_baseline(spec, arch, best_params, batch=batch)
            err = _wall_err_pct(pred["wall_clock_us"], meas["wall_clock_us"])
            errs.append(err)
            per_config.append((cfg_name, bucket, err, pred["wall_clock_us"], meas["wall_clock_us"]))
        return errs, per_config

    train_errs, train_pc = eval_set("TRAIN", train_all)
    val_b_errs, val_b_pc = eval_set("VAL_batch", val_b_all)
    val_s_errs, val_s_pc = eval_set("VAL_size", val_s_all)

    def _report(name, pc):
        print(f"\n=== {name} ===")
        for cfg, bucket, err, wp, wm in pc:
            print(f"  {cfg:40s} [{bucket:>10s}] err={err:>6.1f}% pred={wp:.0f} meas={wm:.0f}")
    _report("TRAIN per-config", train_pc)
    _report("VAL_batch per-config", val_b_pc)
    _report("VAL_size per-config", val_s_pc)

    summary = {
        "per_bucket_fit": {
            b: {k: round(v, 4) for k, v in zip(_PARAMS_PER_BUCKET, per_bucket_theta[b])}
            for b in V6_BUCKET_DEFAULTS.keys()
        },
        "train_mae_pct":      round(statistics.mean(train_errs), 2) if train_errs else None,
        "train_max_pct":      round(max(train_errs), 2) if train_errs else None,
        "val_batch_mae_pct":  round(statistics.mean(val_b_errs), 2) if val_b_errs else None,
        "val_batch_max_pct":  round(max(val_b_errs), 2) if val_b_errs else None,
        "val_size_mae_pct":   round(statistics.mean(val_s_errs), 2) if val_s_errs else None,
        "val_size_max_pct":   round(max(val_s_errs), 2) if val_s_errs else None,
        "n_train":            len(train_all),
        "n_val_batch":        len(val_b_all),
        "n_val_size":         len(val_s_all),
    }
    print(f"\n=== v6 Summary ===")
    print(json.dumps(summary, indent=2))

    return {**best_params, "fit_summary": summary}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="data/calibration/pipe_baseline_per_model.json")
    p.add_argument("--arch", default="arch/ascend_910b4_for_sweep_v2.yaml")
    p.add_argument("--v4-params", default="data/calibration/predict_pipe_params.json")
    p.add_argument("--output", default="data/calibration/predict_pipe_params_v6.json")
    args = p.parse_args()

    result = fit_v6(args.baseline, args.arch, args.v4_params)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
