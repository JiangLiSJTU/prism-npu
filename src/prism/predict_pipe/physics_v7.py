"""v7 physics — SDPA/FlashAttention-aware (Issue #2 v7).

Background:
v6.1 calibrated on eager-attention msprof data, where CANN's auto-fusion
of attention is suboptimal for Qwen3-family. This led to an AIC_QWEN3
bucket with amp_aic=8 + S-scaling.

v7 calibrates on SDPA-path data (production-aligned, since real
deployments use FlashAttention/fused kernels via attn_implementation="sdpa"
or torch_npu.npu_fusion_attention). Under SDPA, Qwen3 family no longer
requires a separate bucket — configs naturally split into BALANCED (short-S)
and AIV_BOUND (long-S) by the same rule used for Llama/Qwen2.5/SmolLM2.

v7 buckets (only 3, no AIC_QWEN3):
  AIC_DECODE  — S == 1
  AIV_BOUND   — d_model >= 700 AND S * batch >= 1024
  BALANCED    — everything else (default)

Validated against 6 Qwen3-sdpa + 7 other measured configs (Phase 1 result).

Naming: v7 keeps the same V6_BUCKET_*_S_alpha keys for back-compat but
DOESN'T USE THEM for non-Qwen3 buckets. S-scaling not needed under SDPA
because attention work scales cleanly through the fused kernel.
"""
from __future__ import annotations
from typing import Literal, Mapping


BucketV7 = Literal["AIC_DECODE", "AIV_BOUND", "BALANCED"]


def classify_bottleneck_v7(spec, batch: int = 1) -> BucketV7:
    """v7 classifier (SDPA-aware) — no AIC_QWEN3 bucket.

    Routes by model compute volume (d_model × layers as a proxy for
    forward-pass GEMM intensity), NOT S*batch. The S*batch threshold in
    v6 mis-classified Qwen3-S256-b1-sdpa (a 596M decoder at S=256-b1
    that's physically similar to its b=4 variant but with S*B=256<1024).

    Calibration thresholds (validated against 13 measured configs):
      tiny / shallow (d<700)        → BALANCED   (BERT, GPT-2 class)
      mid-volume (d≥700, d×L≥12000) → AIV_BOUND  (ModernBERT, Llama, Qwen3, etc.)
      large d_model only (d≥1500)   → AIV_BOUND  (catches shallow-but-wide)
      else                          → BALANCED
    """
    if spec.S == 1:
        return "AIC_DECODE"
    if spec.d_model >= 700:
        # Big model: route by compute volume (d × L), not S*batch
        if spec.d_model * spec.layers >= 12000 or spec.d_model >= 1500:
            return "AIV_BOUND"
    return "BALANCED"


# v7 defaults — initialized from v6.1 non-AIC_QWEN3 buckets (those don't
# change much). Will be refit by fit_v7.py.
V7_BUCKET_DEFAULTS = {
    "AIC_DECODE": {
        "amp_aic":          0.85,
        "amp_aiv":          1.50,
        "nk_mult":          4.70,
    },
    "AIV_BOUND": {
        "amp_aic":          1.5,
        "amp_aiv":          4.5,
        "nk_mult":          6.0,
    },
    "BALANCED": {
        "amp_aic":          2.0,
        "amp_aiv":          3.0,
        "nk_mult":          6.0,
    },
}

V7_BUCKET_BOUNDS = {
    "AIC_DECODE": {
        "amp_aic":          (0.5, 1.5),
        "amp_aiv":          (0.5, 3.0),
        "nk_mult":          (3.0, 6.0),
    },
    "AIV_BOUND": {
        "amp_aic":          (0.5, 4.0),
        "amp_aiv":          (1.5, 8.0),
        "nk_mult":          (3.0, 12.0),
    },
    "BALANCED": {
        "amp_aic":          (0.5, 4.0),
        "amp_aiv":          (0.5, 5.0),
        "nk_mult":          (3.0, 10.0),
    },
}


def get_bucket_params_v7(bucket: BucketV7, params: Mapping[str, float]) -> dict:
    """Resolve v7 per-bucket params. Keys: v7_<BUCKET>_<param>."""
    out = {}
    for k in ("amp_aic", "amp_aiv", "nk_mult"):
        flat_key = f"v7_{bucket}_{k}"
        if flat_key in params:
            out[k] = float(params[flat_key])
        else:
            out[k] = V7_BUCKET_DEFAULTS[bucket][k]
    return out


def high_batch_efficiency_factors(batch: int) -> dict:
    """Empirical per-component 'batch efficiency' factors for high-batch prefill
    (Issue #9 v2 — separate AIC / AIV / n_kern factors).

    v8 物理建模对 batch 是线性外推,但 ATC 在 B≥16 时重新融合 graph,实测
    per-inference 时间亚线性。**重要**:AIC 与 AIV 的 saturation 不同步 ——
    AIC 融合更激进(GEMM 跨 token 易批量 tile fuse),AIV 融合更弱(LN/Softmax/
    GeLU 难以跨 token fuse,vector ops 仍要逐 token 处理)。

    校准数据(Issue #9 Phase 1 + AIV follow-up,Qwen3-prefill-sdpa 2 个真机点,
    用 unscaled v8 prediction 反解 ideal factor):

      Config                     meas / unscaled_pred = ideal_factor
                                  AIC          AIV         n_kern
      Qwen3-prefill-S512-b32     0.093        0.180       0.123
      Qwen3-prefill-S256-b64     0.073        0.209       0.123

    Saturated values(取两点的中点,稍偏保守):
      AIC: 0.10  (issue-9 PR #8 已校准, 留作 baseline)
      AIV: 0.20  ⭐ 关键修正(原误用 0.10 → AIV under 44-52%)
      n_kern: 0.12

    Returns:
      dict with keys 'aic', 'aiv', 'nk', each a float in [floor, 1.0].
      Each factor saturates at its own floor for B≥32; identity 1.0 for B≤8;
      log-space linear interpolation between for B ∈ (8, 32).

    ⚠ 校准来源:仅 Qwen3-prefill-sdpa 2 点。其他家族(Llama / Qwen2.5 / SmolLM2)
    high-batch 数据未采到(ais_bench std::bad_alloc),应假设相同 saturation 规律。
    Confidence: medium for B≥16 (see assign_confidence in predict.py).
    """
    AIC_FLOOR = 0.10
    AIV_FLOOR = 0.20
    NK_FLOOR = 0.12

    if batch <= 8:
        return {"aic": 1.0, "aiv": 1.0, "nk": 1.0}
    if batch >= 32:
        return {"aic": AIC_FLOOR, "aiv": AIV_FLOOR, "nk": NK_FLOOR}

    import math
    t = (math.log2(batch) - 3.0) / (5.0 - 3.0)  # 0 at B=8 (log2=3), 1 at B=32 (log2=5)

    def _interp(floor):
        return 1.0 * (1.0 - t) + floor * t

    return {"aic": _interp(AIC_FLOOR), "aiv": _interp(AIV_FLOOR), "nk": _interp(NK_FLOOR)}


def high_batch_efficiency_factor(batch: int) -> float:
    """Back-compat shim — returns AIC factor only.

    Issue #9 PR #8 had a single uniform factor (= AIC factor in current
    decomposition). External callers that didn't differentiate AIC/AIV are
    safe to call this; predict_v7 now uses ``high_batch_efficiency_factors``
    internally for per-component scaling.
    """
    return high_batch_efficiency_factors(batch)["aic"]


def predict_v7(spec, arch: Mapping[str, float], batch: int,
               params: Mapping[str, float],
               aic_pipes_base: dict, aiv_base_us: float) -> tuple[float, float, int, str]:
    """Apply v7 per-bucket amps. Inputs are physics-base values (no amp).

    Returns: (aic_time_us, aiv_time_us, n_kernels, bucket_name)

    Issue #9 fix (2026-05-24):applies ``high_batch_efficiency_factor(batch)``
    to aic_time / aiv_time / n_kernels to capture ATC kernel re-fusion +
    Cube/AIV saturation at B≥16. Multiplier is 1.0 for B≤8(no regression on
    in-distribution data), saturates to 0.10 for B≥32. See
    ``high_batch_efficiency_factor`` docstring for calibration data.
    """
    bucket = classify_bottleneck_v7(spec, batch)
    bp = get_bucket_params_v7(bucket, params)

    aic_time = max(aic_pipes_base.values()) * bp["amp_aic"]
    aiv_time = aiv_base_us * bp["amp_aiv"]

    # n_kernels base × per-bucket mult
    n_gemm_per_layer = 7 if spec.arch == "decoder" else 8
    n_vec_per_layer = 3
    base = (n_gemm_per_layer + n_vec_per_layer) * spec.layers
    n_kernels = int(base * bp["nk_mult"])

    # Issue #9 v2: per-component high-batch factors(AIC/AIV/n_kern saturate
    # at different rates — see high_batch_efficiency_factors docstring)
    hb = high_batch_efficiency_factors(batch)
    aic_time *= hb["aic"]
    aiv_time *= hb["aiv"]
    n_kernels = int(n_kernels * hb["nk"])

    return aic_time, aiv_time, n_kernels, bucket
