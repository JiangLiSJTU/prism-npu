"""
Analytical pipe-time formulas (parameterized by arch dict).

Migrated from ``.sisyphus/predict_pipe_v0.1.py`` (Windows reviewer prototype).
The v0.1 prototype hardcoded ``ARCH_910B4``; here we take the arch dict as a
parameter so the same formulas work for any NPU with comparable performance
counters.

Each function returns time in microseconds (μs).

Required arch dict keys (validated by ``require_arch_keys()``):
    cube_total_macs, clock_ghz, hbm_bw_gbs, l1_l0_bw_gbs, fixpipe_bw_gbs,
    ub_l1_bw_gbs, aiv_total_throughput
"""
from __future__ import annotations

from typing import Mapping


_REQUIRED_ARCH_KEYS = (
    "cube_total_macs",
    "clock_ghz",
    "hbm_bw_gbs",
    "l1_l0_bw_gbs",
    "fixpipe_bw_gbs",
    "ub_l1_bw_gbs",
    "aiv_total_throughput",
)


def require_arch_keys(arch: Mapping[str, float]) -> None:
    """Raise ValueError if arch dict is missing any required key."""
    missing = [k for k in _REQUIRED_ARCH_KEYS if k not in arch]
    if missing:
        raise ValueError(
            f"arch dict missing required keys: {missing}. "
            f"Required: {list(_REQUIRED_ARCH_KEYS)}"
        )


def aic_mac(total_gemm_ops: float, arch: Mapping[str, float], eta_compute: float = 0.70) -> float:
    """Cube MAC time: Σ(2·M·N·K) / (cube_macs × clock) / η.

    Returns: μs.
    """
    cycles = total_gemm_ops / arch["cube_total_macs"]
    return cycles / (arch["clock_ghz"] * 1e9) * 1e6 / eta_compute


def aic_mte1(activation_read_bytes: float, arch: Mapping[str, float]) -> float:
    """L1↔L0 activation traffic only (weights reused across M-tiles).

    Returns: μs.
    """
    return activation_read_bytes / (arch["l1_l0_bw_gbs"] * 1e9) * 1e6


def aic_mte2(weight_overflow_bytes: float, activation_hbm_bytes: float,
             arch: Mapping[str, float]) -> float:
    """HBM↔L1: weight overflow + activation overflow.

    Returns: μs.
    """
    return (weight_overflow_bytes + activation_hbm_bytes) / (arch["hbm_bw_gbs"] * 1e9) * 1e6


def aic_fixpipe(output_bytes: float, arch: Mapping[str, float]) -> float:
    """L0C→output via fixpipe.

    Returns: μs.
    """
    return output_bytes / (arch["fixpipe_bw_gbs"] * 1e9) * 1e6


def aiv_vec(vector_flops: float, arch: Mapping[str, float]) -> float:
    """Vector ALU time.

    Returns: μs.
    """
    throughput_ops = arch["aiv_total_throughput"] * arch["clock_ghz"] * 1e9
    return vector_flops / throughput_ops * 1e6


def aiv_mte2(intermediate_bytes: float, arch: Mapping[str, float]) -> float:
    """UB↔L1 during vector processing.

    Returns: μs.
    """
    return intermediate_bytes / (arch["ub_l1_bw_gbs"] * 1e9) * 1e6


def aiv_mte3(output_bytes: float, arch: Mapping[str, float]) -> float:
    """UB→output writeback, carried by the MTE3 engine.

    NOTE: MTE3 is a UB-rooted output engine; it is NOT FixPipe (FixPipe is the
    AIC-side L0C→output unit — the AIV has no FixPipe). The model lacks a
    dedicated MTE3 bandwidth knob, so ``ub_l1_bw_gbs`` is used as an empirical
    proxy (UB-side bandwidth, consistent with aiv_mte2). Earlier versions used
    ``fixpipe_bw_gbs`` here — that was a physical mislabel, now corrected.

    Only the v4 legacy path consumes this; v5–v8 split AIV pipes by the
    empirical vec:mte2:mte3 = 1:7:5 ratio and never call this function.
    Returns: μs.
    """
    return output_bytes / (arch["ub_l1_bw_gbs"] * 1e9) * 1e6


# ─────────────────────────────────────────────────────────────────────────
# P2 AIV multi-factor physics (Issue #2): per-kernel overhead + derating
# ─────────────────────────────────────────────────────────────────────────
def aiv_per_kernel_overhead(n_vector_kernels: int,
                            T_init_us: float = 0.04,
                            T_scalar_us: float = 0.15) -> float:
    """Per-kernel fixed cost: instruction init + scalar dispatch overhead.

    Verrocchio (Tang & Wang, JPDC 2023): Init = 40 ns per instruction on
    DaVinci AI Core.  Scalar dispatch adds pipeline setup + register save.

    Returns: μs (total for all vector kernels in one inference).
    """
    return n_vector_kernels * (T_init_us + T_scalar_us)


def eta_repeat(avg_tensor_elements: float,
               repeat_saturation: float = 32768) -> float:
    """Effective Vector throughput derating due to CANN repeat parameter.

    Small tensors → low repeat count → low SIMD lane utilization.
    Zhou et al. (ASPLOS 2025): ``repeat=1`` → 13.54% util; ``repeat=98`` → 100%.
    Modeled as linear ramp saturating at *repeat_saturation* elements.

    Returns: η in [0.05, 1.0].
    """
    return min(1.0, max(avg_tensor_elements / repeat_saturation, 0.05))


def eta_ub_bandwidth(avg_payload_bytes: float,
                     half_bw_threshold: float = 512.0) -> float:
    """UB↔L1 bandwidth derating for small DMA payloads.

    Small transfers pay per-transfer setup cost that dominates; effective BW
    can drop to 30–50 % of peak.  Modeled as a smooth ramp:

        η = 0.3 + 0.7 × min(1, payload / threshold)

    Args:
        avg_payload_bytes: typical single-transfer payload (≈ d_model × 2).
        half_bw_threshold: payload size at which η ≈ 0.65 (halfway from
            floor 0.3 to peak 1.0).  Default 512 B.

    Returns: η in [0.3, 1.0].
    """
    ratio = avg_payload_bytes / half_bw_threshold
    return 0.3 + 0.7 * min(1.0, ratio)


# ─────────────────────────────────────────────────────────────────────────
# P1 empirical correction (Issue #2): CANN tile re-fetching amplification
# ─────────────────────────────────────────────────────────────────────────
def archetype_amplification(weight_mb_proxy: float, S: int) -> float:
    """Empirical aic_time / aiv_time correction by model archetype (Issue #2 P1).

    The v0.1 physics formulas assume "each weight loaded once per inference"
    but CANN tile-by-tile execution reloads weight tiles from HBM many times
    when the per-layer weight matrix exceeds L1 capacity. This produces a
    systematic under-prediction for large models that grows with the model
    size — directly observed on Qwen3 prefill (5-8× under) but not on
    BERT/GPT-2 small models (~1.2× over).

    Calibrated against 23 measured msprof configs (v3, Issue #2 P1 v3):

        BERT-base-S128-b1     (weight_mb≈226, prefill):       ratio = 1.06
        GPT-2-small-S512-b1   (weight_mb≈226, prefill):       ratio = 1.25
        Qwen3-prefill-S256-b1 (weight_mb≈763, S=256):         ratio = 5.19
        Qwen3-prefill-S512-b4 (weight_mb≈763, S=512):         ratio = 5.68
        Qwen3-prefill-S4096-b1 (weight_mb≈763, S=4096):       ratio = 14.16  ← v3
        Qwen3-decode-Skv128   (weight_mb≈763, decode):        ratio = 0.82

    v3 adds the ``large_long`` bucket (S ≥ 4096 + weight_proxy ≥ 600):
    long-context prefill amplifies AIC even more than short-context because
    attention KV-cache re-reads scale with S, and tile re-fetch grows with
    per-kernel work. The targeted bucket eliminates the largest remaining
    outlier (Qwen3-S4096-b1: 61% err in v2 → ~0% in v3).

    Returns a multiplier in [0.85, 14.16] to apply to predicted aic_time.

    Args:
        weight_mb_proxy: rough estimate of per-inference weight read in MB
            (typically ``layers × (4·d_model² + 3·d_model·d_ff) × 2 / 1e6``).
        S: sequence length (S=1 indicates decode regime).
    """
    if S == 1:
        return 0.85   # decode regime: physics slightly over-predicts
    if weight_mb_proxy < 600:
        return 1.15   # small models: physics ~accurate; tiny safety margin
    if S >= 4096:
        return 14.16  # large + long context: CANN tile re-fetch + KV-cache reread
    return 5.5        # large + short context: CANN tile re-fetch dominates


def weight_proxy_mb(layers: int, d_model: int, d_ff: int) -> float:
    """Approximate per-inference weight bytes (MB) for amplification routing.

    Captures the dominant GEMM weights — Q/K/V/O ~ 4·d_model², FFN ~ 3·d_model·d_ff.
    Embedding/LM-head are intentionally excluded (they're a fixed extra ~vocab·d).
    """
    return layers * (4 * d_model**2 + 3 * d_model * d_ff) * 2 / 1e6


def compute_attention_fraction(layers: int, n_heads: int, S: int,
                               total_data_bytes: float) -> float:
    """Fraction of AIV data volume attributed to O(S²) attention softmax.

    The attention score matrix is L × H × S × S × FP16 × 2 (read+write).
    This fraction distinguishes workloads like GPT-2 (56%, amp≈2.0) from
    BERT (24%, amp≈0.9) — the key continuous signal that the 3-bucket
    archetype system misses.

    Args:
        layers: number of transformer layers
        n_heads: number of attention heads (Q heads, not KV heads)
        S: sequence length
        total_data_bytes: total AIV data volume (inter_bytes + output/2 + attn_softmax)

    Returns: fraction in [0, 1]. Returns 0 if total_data_bytes <= 0.
    """
    if total_data_bytes <= 0 or S <= 1:
        return 0.0
    _FP16 = 2
    attn_bytes = float(layers * n_heads * S * S * _FP16 * 2)
    return min(1.0, attn_bytes / total_data_bytes)
