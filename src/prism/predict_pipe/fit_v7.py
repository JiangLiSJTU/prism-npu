"""v7 fit — SDPA-aware per-bucket calibration (Issue #2 v7).

Replaces v6.1's AIC_QWEN3 bucket: instead, Qwen3-sdpa configs distribute
naturally into BALANCED/AIV_BOUND/AIC_DECODE based on (S, batch, d_model).

Per-bucket fit (3 params each): amp_aic, amp_aiv, nk_mult. No S-scaling
needed (validated empirically — SDPA path absorbs attention S² growth
into the fused kernel).

Output: data/calibration/predict_pipe_params_v7.json

Acceptance:
  TRAIN MAE ≤ 15%
  VAL_size MAE ≤ 25%  (cross-architecture within AIV_BOUND)
  VAL_sdpa_long_S err ≤ 30% (Qwen3-S4096-sdpa)
  VAL_sdpa_batch MAE ≤ 25% (Qwen3 b=8 sdpa extrapolation)
"""
from __future__ import annotations
import dataclasses
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

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


def _qwen3_with_S(S: int):
    base = KNOWN_MODELS["Qwen3-prefill-S256-b1"]
    return dataclasses.replace(base, S=S, name=f"Qwen3-prefill-S{S}")


def _load_spec(cfg_name: str, yaml_path: str):
    # Strip -sdpa suffix to look up base spec
    base_name_match = cfg_name.replace("-sdpa", "")
    if base_name_match in KNOWN_MODELS:
        return KNOWN_MODELS[base_name_match]
    base_name = base_name_match.rsplit("-b", 1)[0] + "-b1"
    if base_name in KNOWN_MODELS:
        return KNOWN_MODELS[base_name]
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
        bucket = classify_bottleneck_v7(spec, batch)
        dataset.append((cfg_name, spec, batch, meas, bucket))
    return dataset


def _make_v7_params(per_bucket_theta, fixed):
    p = dict(fixed)
    p["v_model"] = "v7"
    for bucket, theta in per_bucket_theta.items():
        for key, val in zip(_PARAMS_PER_BUCKET, theta):
            p[f"v7_{bucket}_{key}"] = val
    return p


def _wall_err_pct(pred_us, meas_us):
    if not meas_us:
        return 0.0
    return abs(pred_us - meas_us) / meas_us * 100


def _bucket_loss(theta, bucket, configs, arch, other_buckets_params, fixed):
    per_bucket = dict(other_buckets_params)
    per_bucket[bucket] = theta
    params = _make_v7_params(per_bucket, fixed)
    errs = []
    for cfg_name, spec, batch, meas, _b in configs:
        pred = predict_pipe_baseline(spec, arch, params, batch=batch)
        errs.append(_wall_err_pct(pred["wall_clock_us"], meas["wall_clock_us"]))
    return statistics.mean(errs) if errs else 0.0


def fit_v7(baseline_path, arch_yaml, v4_params_path) -> Dict:
    baseline = json.load(open(baseline_path, encoding="utf-8"))
    arch = _arch_dict_from_yaml(arch_yaml)
    v4 = json.load(open(v4_params_path, encoding="utf-8"))

    train = _build_dataset(TRAIN_CONFIGS_V7, baseline)
    val_size = _build_dataset(VAL_SIZE_V7, baseline)
    val_long_s = _build_dataset(VAL_SDPA_LONG_S_V7, baseline)
    val_batch = _build_dataset(VAL_SDPA_BATCH_V7, baseline)
    print(f"Datasets: TRAIN={len(train)}  VAL_size={len(val_size)}  "
          f"VAL_sdpa_long_S={len(val_long_s)}  VAL_sdpa_batch={len(val_batch)}")

    buckets = sorted(set(b for _,_,_,_,b in train))
    print(f"Buckets in TRAIN: {buckets}")
    by_bucket = {b: [c for c in train if c[4] == b] for b in buckets}
    for b, cs in by_bucket.items():
        names = [c[0] for c in cs]
        print(f"  {b}: {len(cs)} configs → {names}")

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

    print("\n=== Per-bucket DE fit ===")
    for bucket in buckets:
        if not by_bucket[bucket]:
            continue
        bounds = [V7_BUCKET_BOUNDS[bucket][k] for k in _PARAMS_PER_BUCKET]
        x0 = per_bucket_theta[bucket]

        def obj(theta):
            return _bucket_loss(theta.tolist(), bucket, by_bucket[bucket],
                                arch, per_bucket_theta, fixed)

        res = scipy.optimize.differential_evolution(
            obj, bounds=bounds, x0=x0, seed=42, maxiter=60, tol=1e-3,
            popsize=15, polish=True, workers=1)
        per_bucket_theta[bucket] = res.x.tolist()
        print(f"  {bucket}: MAE={res.fun:.2f}%  theta={[round(x,3) for x in res.x.tolist()]}")

    best_params = _make_v7_params(per_bucket_theta, fixed)

    def eval_set(name, ds):
        errs, per_config = [], []
        for cfg, spec, batch, meas, bucket in ds:
            pred = predict_pipe_baseline(spec, arch, best_params, batch=batch)
            err = _wall_err_pct(pred["wall_clock_us"], meas["wall_clock_us"])
            errs.append(err)
            per_config.append((cfg, bucket, err, pred["wall_clock_us"], meas["wall_clock_us"]))
        return errs, per_config

    train_errs, train_pc = eval_set("TRAIN", train)
    val_size_errs, val_size_pc = eval_set("VAL_size", val_size)
    val_long_errs, val_long_pc = eval_set("VAL_sdpa_long_S", val_long_s)
    val_batch_errs, val_batch_pc = eval_set("VAL_sdpa_batch", val_batch)

    def _report(name, pc):
        print(f"\n=== {name} ===")
        for cfg, bucket, err, wp, wm in pc:
            print(f"  {cfg:42s} [{bucket:>10s}] err={err:>6.1f}% pred={wp:>9.0f} meas={wm:>9.0f}")
    _report("TRAIN", train_pc)
    _report("VAL_size", val_size_pc)
    _report("VAL_sdpa_long_S", val_long_pc)
    _report("VAL_sdpa_batch", val_batch_pc)

    summary = {
        "per_bucket_fit": {
            b: {k: round(v, 4) for k, v in zip(_PARAMS_PER_BUCKET, per_bucket_theta[b])}
            for b in V7_BUCKET_DEFAULTS.keys()
        },
        "train_mae_pct":           round(statistics.mean(train_errs), 2) if train_errs else None,
        "train_max_pct":           round(max(train_errs), 2) if train_errs else None,
        "val_size_mae_pct":        round(statistics.mean(val_size_errs), 2) if val_size_errs else None,
        "val_size_max_pct":        round(max(val_size_errs), 2) if val_size_errs else None,
        "val_sdpa_long_s_err_pct": round(val_long_errs[0], 2) if val_long_errs else None,
        "val_sdpa_batch_mae_pct":  round(statistics.mean(val_batch_errs), 2) if val_batch_errs else None,
        "val_sdpa_batch_max_pct":  round(max(val_batch_errs), 2) if val_batch_errs else None,
        "n_train":                 len(train),
        "n_val_size":              len(val_size),
        "n_val_sdpa_long_s":       len(val_long_s),
        "n_val_sdpa_batch":        len(val_batch),
    }
    print(f"\n=== v7 Summary ===")
    print(json.dumps(summary, indent=2))

    return {**best_params, "fit_summary": summary}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="data/calibration/pipe_baseline_per_model.json")
    p.add_argument("--arch",     default="arch/ascend_910b4_for_sweep_v2.yaml")
    p.add_argument("--v4-params",default="data/calibration/predict_pipe_params.json")
    p.add_argument("--output",   default="data/calibration/predict_pipe_params_v7.json")
    args = p.parse_args()
    result = fit_v7(args.baseline, args.arch, args.v4_params)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
