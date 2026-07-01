"""
Fit host_gap + kernel_gap constants from msprof pipe baseline measurements.

Two empirical interaction terms that are NOT derivable from arch+model physics:

- ``kernel_gap`` ≈ K0 × n_kernels (OLS through origin)
- ``host_gap`` ≈ constant per regime (prefill ≈ 13 ms, decode ≈ 0.2 ms)

The Windows reviewer's v0.1 prototype documented:
  K0 ≈ 1.86 μs/kernel, MAE 14.3%
  H_prefill ≈ 13,424 μs (MAE 8.4%)
  H_decode ≈ 204 μs (MAE 0%)

These are the **interaction constants between CANN runtime and model layout** —
they characterize the host scheduling overhead per kernel and the steady-state
host_gap floor per inference. They are arch-invariant under the v3 model and
get baked into the predict step.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

from .model_spec import KNOWN_MODELS, ModelSpec, compute_vector_ops, compute_gemm_ops, estimate_n_vector_kernels
from . import physics


def _is_decode(cfg_name: str) -> bool:
    """Heuristic: a config is a 'decode' regime if its name says so."""
    name_lower = cfg_name.lower()
    return ("decode" in name_lower) or ("Min" in cfg_name)


def _is_measured(cfg: Mapping) -> bool:
    """True if the config has real msprof data (not an estimate/inherited placeholder)."""
    source = cfg.get("source", "")
    return not source.startswith(("estimated", "inherited"))


def fit_host_gap(pipe_baseline: Mapping[str, Mapping],
                 known_specs: Mapping[str, ModelSpec] = None) -> Tuple[float, float, List[float]]:
    """Fit host_gap as a constant per regime (prefill vs decode).

    Returns:
        (H_prefill_us, H_decode_us, training_errors_pct)
    """
    if known_specs is None:
        known_specs = KNOWN_MODELS

    prefill_hg: List[float] = []
    decode_hg: List[float] = []

    for cfg_name, cfg in pipe_baseline.items():
        if cfg_name not in known_specs or not _is_measured(cfg):
            continue
        hg = cfg["host_gap_us"]
        if _is_decode(cfg_name):
            decode_hg.append(hg)
        elif cfg["n_kernels_per_inf"] > 500:   # multi-kernel = prefill-like
            prefill_hg.append(hg)

    H_prefill = sum(prefill_hg) / len(prefill_hg) if prefill_hg else 13000.0
    H_decode = sum(decode_hg) / len(decode_hg) if decode_hg else 200.0

    # Compute training errors
    errors: List[float] = []
    for cfg_name, cfg in pipe_baseline.items():
        if cfg_name not in known_specs or not _is_measured(cfg):
            continue
        actual = cfg["host_gap_us"]
        pred = H_decode if _is_decode(cfg_name) else H_prefill
        if actual > 0:
            errors.append(abs(pred - actual) / actual * 100)

    return H_prefill, H_decode, errors


def _is_sdpa(cfg_name: str) -> bool:
    """SDPA configs are suffixed with '-sdpa' by convention (Issue #2 v7+)."""
    return cfg_name.endswith("-sdpa")


def _is_qwen3_family(cfg_name: str) -> bool:
    return "Qwen3" in cfg_name


def fit_host_gap_sdpa(pipe_baseline: Mapping[str, Mapping],
                      known_specs: Mapping[str, ModelSpec] = None) -> Tuple[Dict[str, float], List[float], Dict[str, float]]:
    """Issue #11 — fit per-family multivariate host_gap model for SDPA path.

    The existing constant `H_prefill = 13,424 μs` fits eager-attention configs
    but systematically under-predicts SDPA-path host_gap by 1.1× to 47×:

      SDPA non-Qwen3 (5 configs): host_gap 15k - 57k μs (≈ 30k mean, 4× ratio)
      SDPA Qwen3 normal-batch (5 configs): 47k - 65k μs (~ 60k, 4-5× ratio)
      SDPA Qwen3 high-batch (B≥32, 2 configs): 527k - 632k μs (40-47× ratio)

    Single constant cannot fit all 3 regimes. Pooled OLS gives MAE 274%.
    Per-family multivariate fit gives ≤5% MAE for Qwen3, ≤20% for others.

    Model (calibrated on 12 SDPA configs from msprof PipeUtilization):

      Qwen3 SDPA:    H = α_q + β_q × n_kernels + γ_q × (B × S)
      Other SDPA:    H = α_o + γ_o × (B × S)   (no n_kern feature — small dataset)

    Returns: (params_dict, training_errors_pct, debug_info)

    The dispatch is performed in ``predict.predict_pipe_baseline`` via the new
    ``attn_impl`` parameter:
      attn_impl="eager" (default) → uses H_prefill_us (existing constant)
      attn_impl="sdpa" → uses per-family multivariate fit
    """
    if known_specs is None:
        known_specs = KNOWN_MODELS
    import numpy as np  # used only here; widely available

    rows_q3, rows_other = [], []
    for cfg_name, cfg in pipe_baseline.items():
        if not _is_sdpa(cfg_name) or not _is_measured(cfg) or _is_decode(cfg_name):
            continue
        hg = cfg.get("host_gap_us", 0)
        nk = cfg.get("n_kernels_per_inf", 0)
        if hg <= 0 or nk <= 0:
            continue
        # Parse B and S from config name (e.g. "Qwen3-prefill-S256-b64-sdpa")
        import re
        mS = re.search(r"-S(\d+)", cfg_name); mB = re.search(r"-b(\d+)", cfg_name)
        if not (mS and mB):
            continue
        S = int(mS.group(1)); B = int(mB.group(1)); BS = B * S
        target = (rows_q3 if _is_qwen3_family(cfg_name) else rows_other)
        target.append((cfg_name, nk, BS, hg))

    out: Dict[str, float] = {}
    errors: List[float] = []
    debug: Dict[str, float] = {"n_qwen3": len(rows_q3), "n_other": len(rows_other)}

    # Qwen3 fit (3 params: α + β·nk + γ·BS) — needs ≥4 points
    if len(rows_q3) >= 4:
        X = np.array([[1.0, r[1], r[2]] for r in rows_q3], dtype=float)
        y = np.array([r[3] for r in rows_q3], dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        out["H_prefill_sdpa_qwen3_alpha"] = float(coef[0])
        out["H_prefill_sdpa_qwen3_beta_nk"] = float(coef[1])
        out["H_prefill_sdpa_qwen3_gamma_BS"] = float(coef[2])
        pred = X @ coef
        per_err = [abs(p - r[3]) / r[3] * 100 for p, r in zip(pred, rows_q3) if r[3] > 0]
        debug["qwen3_mae_pct"] = sum(per_err) / len(per_err) if per_err else 0.0
        debug["qwen3_max_pct"] = max(per_err) if per_err else 0.0
        errors.extend(per_err)
    else:
        out["H_prefill_sdpa_qwen3_alpha"] = 60000.0  # safe default ≈ mean of normal-batch
        out["H_prefill_sdpa_qwen3_beta_nk"] = 0.0
        out["H_prefill_sdpa_qwen3_gamma_BS"] = 30.0
        debug["qwen3_mae_pct"] = float("nan")

    # Other-family SDPA fit (2 params: α + γ·BS) — needs ≥3 points
    if len(rows_other) >= 3:
        X = np.array([[1.0, r[2]] for r in rows_other], dtype=float)
        y = np.array([r[3] for r in rows_other], dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        out["H_prefill_sdpa_other_alpha"] = float(coef[0])
        out["H_prefill_sdpa_other_gamma_BS"] = float(coef[1])
        pred = X @ coef
        per_err = [abs(p - r[3]) / r[3] * 100 for p, r in zip(pred, rows_other) if r[3] > 0]
        debug["other_mae_pct"] = sum(per_err) / len(per_err) if per_err else 0.0
        debug["other_max_pct"] = max(per_err) if per_err else 0.0
        errors.extend(per_err)
    else:
        out["H_prefill_sdpa_other_alpha"] = 30000.0  # safe default ≈ mean of measured 5
        out["H_prefill_sdpa_other_gamma_BS"] = 0.0
        debug["other_mae_pct"] = float("nan")

    return out, errors, debug


def fit_kernel_gap(pipe_baseline: Mapping[str, Mapping],
                   known_specs: Mapping[str, ModelSpec] = None
                   ) -> Tuple[float, List[int], List[float], List[float]]:
    """Fit ``kernel_gap = K0 × n_kernels`` by OLS through origin.

    Returns:
        (K0_us_per_kernel, X_n_kernels, Y_kernel_gaps, training_errors_pct)
    """
    if known_specs is None:
        known_specs = KNOWN_MODELS

    X: List[int] = []
    Y: List[float] = []
    cfg_names: List[str] = []

    for cfg_name, cfg in pipe_baseline.items():
        if cfg_name not in known_specs or "kernel_gap_us" not in cfg:
            continue
        X.append(int(cfg["n_kernels_per_inf"]))
        Y.append(float(cfg["kernel_gap_us"]))
        cfg_names.append(cfg_name)

    if len(X) < 2:
        return 0.0, [], [], []

    sum_x2 = sum(x * x for x in X)
    sum_xy = sum(x * y for x, y in zip(X, Y))
    K0 = sum_xy / sum_x2 if sum_x2 > 0 else 0.0

    errors = [abs(y - K0 * x) / y * 100 if y > 0 else 0.0 for y, x in zip(Y, X)]
    return K0, X, Y, errors


def leave_one_model_out_cv(pipe_baseline: Mapping[str, Mapping],
                           known_specs: Mapping[str, ModelSpec] = None
                           ) -> Dict[str, Dict]:
    """Leave-one-model-family-out cross-validation.

    For each unique model family in known_specs (e.g. BERT-base, GPT-2-small,
    Qwen3-0.6B), refit (K0, H_prefill, H_decode) on the remaining families and
    measure prediction error on the held-out one.

    Returns:
        Dict keyed by held-out model name, with fit params + per-config test errors.
    """
    if known_specs is None:
        known_specs = KNOWN_MODELS

    measured_families = sorted({
        known_specs[n].name for n in known_specs
        if n in pipe_baseline and _is_measured(pipe_baseline[n])
    })

    results: Dict[str, Dict] = {}
    for held_out in measured_families:
        train_configs: Dict[str, Mapping] = {}
        test_configs: Dict[str, Mapping] = {}
        for cfg_name, cfg in pipe_baseline.items():
            if cfg_name not in known_specs:
                continue
            if known_specs[cfg_name].name == held_out:
                test_configs[cfg_name] = cfg
            else:
                train_configs[cfg_name] = cfg

        H_prefill, H_decode, train_hg_errs = fit_host_gap(train_configs, known_specs)
        K0, _, _, train_kg_errs = fit_kernel_gap(train_configs, known_specs)

        test_results: List[Dict] = []
        for cfg_name, cfg in test_configs.items():
            n_kernels = cfg["n_kernels_per_inf"]
            actual_hg = cfg["host_gap_us"]
            pred_hg = H_decode if _is_decode(cfg_name) else H_prefill
            hg_err = abs(pred_hg - actual_hg) / actual_hg * 100 if actual_hg > 0 else 0.0
            pred_kg = K0 * n_kernels
            actual_kg = cfg.get("kernel_gap_us", 0)
            kg_err = abs(pred_kg - actual_kg) / actual_kg * 100 if actual_kg > 0 else 0.0
            test_results.append({
                "config": cfg_name,
                "host_gap_pred_us": pred_hg,
                "host_gap_actual_us": actual_hg,
                "host_gap_err_pct": hg_err,
                "kernel_gap_pred_us": pred_kg,
                "kernel_gap_actual_us": actual_kg,
                "kernel_gap_err_pct": kg_err,
            })

        results[held_out] = {
            "fit_H_prefill_us": H_prefill,
            "fit_H_decode_us": H_decode,
            "fit_K0_us_per_kernel": K0,
            "train_host_gap_mae_pct": (sum(train_hg_errs) / len(train_hg_errs)
                                       if train_hg_errs else 0.0),
            "train_kernel_gap_mae_pct": (sum(train_kg_errs) / len(train_kg_errs)
                                         if train_kg_errs else 0.0),
            "test_configs": test_results,
        }
    return results


def fit_aiv_params(pipe_baseline: Mapping[str, Mapping],
                   arch: Mapping[str, float],
                   known_specs: Mapping[str, ModelSpec] = None,
                   ) -> Dict[str, float]:
    """Fit 5 AIV empirical model parameters from msprof pipe baselines (v4 Method B).

    Uses brute-force grid search over physically-constrained ranges,
    minimizing MAE against measured ``aiv_time_us`` across all known configs.

    The v4 (Method B) model replaces the 3-bucket discrete amplification
    with a **continuous** function:

        aiv_time = (n_vk × C_kernel + data_MB × C_data) × amp

    where ``amp`` for prefill is:

        amp = max(0.1, a0 + a1 × attn_frac + a2 × (w_proxy/1000)²)

    and ``attn_frac`` = O(S²) attention softmax bytes / total AIV data bytes.

    This resolves the key failure of the 3-bucket system: GPT-2 (attn_frac=0.56)
    gets amp≈2.0 while BERT (attn_frac=0.24) gets amp≈0.9, instead of both
    being lumped into the same "small" bucket.

    Parameters fitted:
        aiv_C_kernel_us:   per-kernel fixed cost (μs/kernel)
        aiv_C_data_us:     data-proportional cost (μs/MB)
        aiv_amp_a0:        intercept for continuous amp function
        aiv_amp_a1:        attn_frac coefficient
        aiv_amp_a2:        w_proxy/1000 coefficient
        aiv_amp_decode:    decode-specific constant

    Returns:
        Dict with best-fit parameters + ``aiv_mae_pct`` + ``aiv_per_config``.
    """
    if known_specs is None:
        known_specs = KNOWN_MODELS

    _FP16 = 2

    # Collect measured configs with aiv_time_us > 0
    measured: List[Tuple[str, ModelSpec, float]] = []
    for cfg_name, cfg in pipe_baseline.items():
        if cfg_name not in known_specs:
            continue
        aiv_meas = cfg.get("aiv_time_us", 0)
        if aiv_meas <= 0:
            continue
        if cfg.get("source", "").startswith(("estimated", "inherited")):
            continue
        measured.append((cfg_name, known_specs[cfg_name], float(aiv_meas)))

    if len(measured) < 2:
        # Not enough data; return priors
        return {
            "aiv_C_kernel_us": 16.0,
            "aiv_C_data_us": 3.0,
            "aiv_amp_a0": -0.2,
            "aiv_amp_a1": 4.0,
            "aiv_amp_a2": 14.0,
            "aiv_amp_decode": 1.5,
            "aiv_mae_pct": 0.0,
            "aiv_n_configs": len(measured),
        }

    # Pre-compute features for each config (avoid repeated recomputation)
    config_features: List[Tuple[float, float, int, float, bool]] = []
    # Each tuple: (data_MB, attn_frac, n_vk, w_proxy_scaled, is_decode)
    for cfg_name, spec, _ in measured:
        batch = int(cfg_name.split("-b")[-1]) if "-b" in cfg_name else 1
        f = float(batch)
        _, inter_bytes = compute_vector_ops(spec)
        _, _, _, output_b = compute_gemm_ops(spec)
        attn_softmax_bytes = float(spec.layers * spec.n_heads
                                   * spec.S * spec.S * _FP16 * 2) * f
        data_bytes = inter_bytes * f + output_b * 0.5 * f + attn_softmax_bytes
        data_MB = data_bytes / 1e6
        # attn_frac is a model geometry property — pass per-batch data bytes
        attn_frac = physics.compute_attention_fraction(
            spec.layers, spec.n_heads, spec.S, data_bytes / f)
        n_vk = estimate_n_vector_kernels(spec)
        w_proxy = physics.weight_proxy_mb(spec.layers, spec.d_model, spec.d_ff)
        config_features.append((data_MB, attn_frac, n_vk,
                                (w_proxy / 1000.0) ** 2,  # squared: tile re-fetch scales quadratically
                                spec.S == 1))

    # Grid search — 6D: C_kernel × C_data × a0 × a1 × a2 × amp_decode
    # Centered around fine-grid optimum (Ck=16, Cd=3, a0=-0.2, a1=4.0, a2=14.0, amp_d=1.5)
    # which achieves 4.9% AIV MAE on 6 configs.
    # a2 multiplies (w_proxy/1000)² — range 0 to 0.58 for Qwen3, ~0.05 for BERT
    grid_C_kernel = [8.0, 12.0, 16.0, 22.0, 30.0]
    grid_C_data = [1.0, 2.0, 3.0, 5.0, 8.0]
    grid_a0 = [-0.5, -0.2, 0.0, 0.3, 0.6]
    grid_a1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    grid_a2 = [6.0, 9.0, 11.0, 14.0, 17.0, 20.0, 24.0]
    grid_amp_decode = [0.5, 1.0, 1.5, 2.0, 3.0]

    best_mae = float("inf")
    best_params: Dict[str, float] = {}

    for c_kernel in grid_C_kernel:
        for c_data in grid_C_data:
            for a0 in grid_a0:
                for a1 in grid_a1:
                    for a2 in grid_a2:
                        for amp_d in grid_amp_decode:
                            errs: List[float] = []
                            for i, (cfg_name, spec, aiv_meas) in enumerate(measured):
                                data_MB, attn_frac, n_vk, wp_sq, is_dec = config_features[i]
                                if is_dec:
                                    amp = amp_d
                                else:
                                    amp = max(0.1, a0 + a1 * attn_frac + a2 * wp_sq)
                                pred = (n_vk * c_kernel + data_MB * c_data) * amp
                                errs.append(abs(pred - aiv_meas) / aiv_meas * 100)
                            mae = sum(errs) / len(errs)
                            if mae < best_mae:
                                best_mae = mae
                                best_params = {
                                    "aiv_C_kernel_us": c_kernel,
                                    "aiv_C_data_us": c_data,
                                    "aiv_amp_a0": a0,
                                    "aiv_amp_a1": a1,
                                    "aiv_amp_a2": a2,
                                    "aiv_amp_decode": amp_d,
                                }

    # Compute per-config errors with best params
    per_config: List[Dict] = []
    for i, (cfg_name, spec, aiv_meas) in enumerate(measured):
        data_MB, attn_frac, n_vk, wp_sq, is_dec = config_features[i]
        if is_dec:
            amp = best_params["aiv_amp_decode"]
        else:
            amp = max(0.1, best_params["aiv_amp_a0"]
                      + best_params["aiv_amp_a1"] * attn_frac
                      + best_params["aiv_amp_a2"] * wp_sq)
        pred = (n_vk * best_params["aiv_C_kernel_us"]
                + data_MB * best_params["aiv_C_data_us"]) * amp
        err = abs(pred - aiv_meas) / aiv_meas * 100
        per_config.append({
            "config": cfg_name,
            "aiv_pred_us": round(pred, 1),
            "aiv_meas_us": round(aiv_meas, 1),
            "aiv_err_pct": round(err, 1),
            "attn_frac": round(attn_frac, 3),
            "amp_computed": round(amp, 3),
        })

    best_params["aiv_mae_pct"] = round(best_mae, 1)
    best_params["aiv_n_configs"] = len(measured)
    best_params["aiv_per_config"] = per_config
    return best_params


def fit_all_and_save(pipe_baseline_path: Path | str,
                     output_path: Path | str,
                     known_specs: Mapping[str, ModelSpec] = None,
                     arch: Mapping[str, float] | None = None) -> Dict:
    """Fit K0, H_prefill, H_decode + AIV multi-factor params and persist to JSON.

    Output schema::

        {
          "K0_us_per_kernel": 1.86,
          "H_prefill_us": 13424,
          "H_decode_us": 204,
          "aiv_C_kernel_us": 13.5,
          "aiv_C_data_us": 11.46,
          "aiv_amp_a0": -0.2,
          "aiv_amp_a1": 4.0,
          "aiv_amp_a2": 14.0,
          "aiv_amp_decode": 1.5,
          "training": {"host_gap_mae_pct": 8.4, "kernel_gap_mae_pct": 14.3, ...},
          "aiv_training": {"mae_pct": ..., "n_configs": ..., "per_config": [...]},
          "loo_cv": {<held-out family>: {...}}
        }
    """
    if known_specs is None:
        known_specs = KNOWN_MODELS

    with open(pipe_baseline_path, encoding="utf-8") as f:
        baseline_doc = json.load(f)
    configs = baseline_doc.get("configs", baseline_doc)

    H_prefill, H_decode, hg_errs = fit_host_gap(configs, known_specs)
    K0, _, _, kg_errs = fit_kernel_gap(configs, known_specs)
    loo = leave_one_model_out_cv(configs, known_specs)

    # Issue #11 — multivariate host_gap fit for SDPA path (per-family)
    sdpa_params, sdpa_errs, sdpa_debug = fit_host_gap_sdpa(configs, known_specs)

    result: Dict = {
        "K0_us_per_kernel": K0,
        "H_prefill_us": H_prefill,
        "H_decode_us": H_decode,
        **sdpa_params,  # Issue #11: H_prefill_sdpa_qwen3_{alpha,beta_nk,gamma_BS} + _other_{alpha,gamma_BS}
        "training": {
            "host_gap_mae_pct": sum(hg_errs) / len(hg_errs) if hg_errs else 0.0,
            "kernel_gap_mae_pct": sum(kg_errs) / len(kg_errs) if kg_errs else 0.0,
            "n_configs": len([c for c in configs.values()
                              if _is_measured(c) and c.get("kernel_gap_us")]),
            "host_gap_sdpa_mae_pct": sum(sdpa_errs) / len(sdpa_errs) if sdpa_errs else 0.0,
            "host_gap_sdpa_debug": sdpa_debug,
        },
        "loo_cv": loo,
    }

    # AIV multi-factor fit (Issue #2 P2): requires arch dict for physics formulas
    if arch is not None:
        aiv = fit_aiv_params(configs, arch, known_specs)
        # Merge top-level fittable params (v4 Method B: continuous amp)
        for key in ("aiv_C_kernel_us", "aiv_C_data_us",
                     "aiv_amp_a0", "aiv_amp_a1", "aiv_amp_a2", "aiv_amp_decode"):
            result[key] = aiv[key]
        result["aiv_training"] = {
            "mae_pct": aiv["aiv_mae_pct"],
            "n_configs": aiv["aiv_n_configs"],
            "per_config": aiv.get("aiv_per_config", []),
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
