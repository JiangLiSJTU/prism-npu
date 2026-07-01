#!/usr/bin/env python3
"""Issue #9 Phase 2a — Evaluate v8 prediction error per-component on high-batch configs.

Loads current v8 params + ingested high-batch baseline configs (`-b32-sdpa` /
`-b64-sdpa`), runs the v8 physics path (predict_pipe_baseline), and reports
signed component errors vs msprof measured. Includes low-batch Qwen3 sdpa
configs as in-distribution sanity reference.

Smoking gun for Issue #9:
    p_nk / m_nk ≈ 1665 / 204 = 8.16×
    aic_err   ≈ +1000% (massive over-prediction)
    aiv_err   ≈ +400%  (massive over-prediction)
    wall_err  ≈ -20%   (negative because H_prefix=13424μs dominates pred wall,
                        masks the AIC/AIV over-pred when batch=1-8 was fitted)

After Phase 2b fix, re-run this script — high-batch aic_err / aiv_err should
drop to <30% (matching the in-distribution band).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import prism.predict_pipe.fit_v8 as fv8  # noqa: E402
from prism.predict_pipe.fit_v8 import _arch_dict_from_yaml, _load_spec  # noqa: E402
from prism.predict_pipe.physics_v7 import classify_bottleneck_v7  # noqa: E402
from prism.predict_pipe.predict import predict_pipe_baseline  # noqa: E402

BASELINE_PATH = REPO / "data" / "calibration" / "pipe_baseline_per_model.json"
PARAMS_PATH = REPO / "data" / "calibration" / "predict_pipe_params_v8.json"
ARCH_YAML = REPO / "arch" / "ascend_910b4_for_sweep_v2.yaml"

# (config_name, model_yaml, batch, split_label) — ★ marks new Issue #9 collection.
CASES: List[Tuple[str, str, int, str]] = [
    # ★ Issue #9 new high-batch configs (Phase 1 collection)
    ("Qwen3-prefill-S512-b32-sdpa", "models/regime/qwen3_0.6b_prefill_S512.yaml", 32, "★ NEW-b32"),
    ("Qwen3-prefill-S256-b64-sdpa", "models/regime/qwen3_0.6b_prefill_S256.yaml", 64, "★ NEW-b64"),
    # In-distribution Qwen3 sdpa (TRAIN + VAL) for sanity reference
    ("Qwen3-prefill-S256-b1-sdpa",  "models/regime/qwen3_0.6b_prefill_S256.yaml",  1, "TRAIN"),
    ("Qwen3-prefill-S256-b4-sdpa",  "models/regime/qwen3_0.6b_prefill_S256.yaml",  4, "TRAIN"),
    ("Qwen3-prefill-S256-b8-sdpa",  "models/regime/qwen3_0.6b_prefill_S256.yaml",  8, "VAL_batch"),
    ("Qwen3-prefill-S512-b4-sdpa",  "models/regime/qwen3_0.6b_prefill_S512.yaml",  4, "TRAIN"),
    ("Qwen3-prefill-S512-b8-sdpa",  "models/regime/qwen3_0.6b_prefill_S512.yaml",  8, "VAL_batch"),
    ("Qwen3-prefill-S4096-b1-sdpa", "models/regime/qwen3_0.6b_prefill_S4096.yaml", 1, "VAL_long_S"),
]


def main() -> int:
    baseline = json.load(BASELINE_PATH.open(encoding="utf-8"))
    params = json.load(PARAMS_PATH.open(encoding="utf-8"))
    arch = _arch_dict_from_yaml(str(ARCH_YAML))
    fv8.arch_dict = arch

    header = (f"{'config':<46}  {'split':<10}  {'bucket':<10}  "
              f"{'wall':>9}  {'aic':>9}  {'aiv':>9}  {'p_nk':>6}/{'m_nk':<6}")
    print(header)
    print("-" * len(header))

    n_high_batch_bad = 0
    for cfg, yaml, batch, split in CASES:
        if cfg not in baseline["configs"]:
            print(f"{cfg:<46}  {split:<10}  (missing in baseline)")
            continue
        spec = _load_spec(cfg, yaml)
        if spec is None:
            print(f"{cfg:<46}  {split:<10}  (could not load spec)")
            continue
        meas = baseline["configs"][cfg]
        attn_impl = "sdpa" if cfg.endswith("-sdpa") else "eager"
        pred = predict_pipe_baseline(spec, arch, params, batch=batch, attn_impl=attn_impl)
        bucket = classify_bottleneck_v7(spec, batch)

        def err_pct(p, m):
            return 100 * (p - m) / max(m, 1)

        we = err_pct(pred["wall_clock_us"], meas["wall_clock_us"])
        ae = err_pct(pred["aic_time_us"],   meas["aic_time_us"])
        ve = err_pct(pred["aiv_time_us"],   meas["aiv_time_us"])
        print(f"{cfg:<46}  {split:<10}  {bucket:<10}  "
              f"{we:>+8.1f}%  {ae:>+8.1f}%  {ve:>+8.1f}%  "
              f"{pred['n_kernels_per_inf']:>6}/{meas['n_kernels_per_inf']:<6}")

        if split.startswith("★") and (abs(ae) > 100 or abs(ve) > 100):
            n_high_batch_bad += 1

    print()
    if n_high_batch_bad > 0:
        print(f"⚠ {n_high_batch_bad} high-batch config(s) have |aic_err| or |aiv_err| > 100%")
        print("  (smoking gun for Issue #9 — n_kernels not batch-aware → ATC fusion regime ignored)")
    else:
        print("✓ all high-batch configs within 100% on AIC/AIV — fix may be working")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
