"""
ModelSpec — GEMM-level model description for analytical pipe prediction.

A ``ModelSpec`` captures the transformer hyperparameters needed to compute
total GEMM ops, activation/weight bytes, and vector FLOPs. From these you
can derive expected pipe times without any msprof measurement.

YAML format (extends ``models/regime/*.yaml``)::

    name: ModernBERT-base-S4096-b1
    arch: encoder                       # encoder | decoder
    layers: 22
    # ... existing regime fields (ops_b1, bytes_total, ...)
    gemm_spec:                          # OPTIONAL — required by predict_pipe
      S: 4096
      d_model: 768
      d_ff: 1152                        # for GLU, this is the *single* projection N
                                        # (gate_proj and up_proj each [S,d_model]→[S,d_ff])
      n_heads: 12
      n_kv_heads: 0                     # 0 = MHA; > 0 = GQA with this many KV heads
      d_head: 64
      vocab: 50368
      ffn_type: glu                     # standard | glu | swiglu

The Windows reviewer's prototype hard-coded ``ModelSpec`` instances; this
module also exposes a ``KNOWN_MODELS`` registry for the 6 baseline configs
that have msprof measurements (needed for leave-one-out CV in fit.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import yaml

# Bytes per FP16 element
_FP16_BYTES = 2


@dataclass
class ModelSpec:
    """GEMM-level transformer spec for analytical pipe prediction."""
    name: str
    arch: str          # "encoder" | "decoder"
    layers: int
    S: int             # sequence length (or 1 for decode)
    d_model: int
    d_ff: int
    n_heads: int
    n_kv_heads: int    # 0 = MHA; > 0 = GQA with this many KV heads
    d_head: int
    vocab: int
    ffn_type: str = "standard"   # "standard" | "glu" | "swiglu"
    note: str = ""

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ModelSpec":
        """Load a ModelSpec from a regime YAML with ``gemm_spec:`` block.

        Raises:
            FileNotFoundError: path does not exist
            KeyError: missing required top-level field or ``gemm_spec`` block
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)

        if "gemm_spec" not in doc:
            raise KeyError(
                f"{path}: missing 'gemm_spec:' block required by predict_pipe. "
                f"See docs/tutorials/05_predict_new_model.md for the schema."
            )
        gs = doc["gemm_spec"]
        return cls(
            name=str(doc["name"]),
            arch=str(doc["arch"]),
            layers=int(doc["layers"]),
            S=int(gs["S"]),
            d_model=int(gs["d_model"]),
            d_ff=int(gs["d_ff"]),
            n_heads=int(gs["n_heads"]),
            n_kv_heads=int(gs.get("n_kv_heads", 0)),
            d_head=int(gs["d_head"]),
            vocab=int(gs["vocab"]),
            ffn_type=str(gs.get("ffn_type", "standard")),
            note=str(doc.get("note", "")).strip(),
        )


def compute_gemm_ops(spec: ModelSpec) -> Tuple[float, float, float, float]:
    """Compute total GEMM ops + activation/weight/output bytes per inference (B=1).

    Returns:
        (total_ops, activation_read_bytes, weight_bytes, output_bytes)

    Bytes assume FP16. ``activation_read_bytes`` is the L1↔L0 read volume
    (input side); ``weight_bytes`` is the per-inference HBM read traffic if
    weights cannot fit on-chip (caller computes overflow); ``output_bytes`` is
    L0C→fixpipe→output write traffic.
    """
    L = spec.layers
    S = spec.S
    d = spec.d_model
    d_ff = spec.d_ff
    H = spec.n_heads
    d_h = spec.d_head
    n_kv = spec.n_kv_heads if spec.n_kv_heads > 0 else H

    per_layer_ops = 0.0
    per_layer_weight = 0.0
    per_layer_act_read = 0.0
    per_layer_output = 0.0

    # Q projection: [S, d_model] × [d_model, H × d_head]
    M, N, K = S, H * d_h, d
    per_layer_ops += 2 * M * N * K
    per_layer_weight += (K * N) * _FP16_BYTES
    per_layer_act_read += (M * K) * _FP16_BYTES
    per_layer_output += (M * N) * _FP16_BYTES

    # K projection: [S, d_model] × [d_model, n_kv × d_head]
    M, N, K = S, n_kv * d_h, d
    per_layer_ops += 2 * M * N * K
    per_layer_weight += (K * N) * _FP16_BYTES
    per_layer_act_read += (M * K) * _FP16_BYTES
    per_layer_output += (M * N) * _FP16_BYTES

    # V projection: same dims as K
    per_layer_ops += 2 * M * N * K
    per_layer_weight += (K * N) * _FP16_BYTES
    per_layer_act_read += (M * K) * _FP16_BYTES
    per_layer_output += (M * N) * _FP16_BYTES

    # QK^T attention: [H, S, d_h] × [H, d_h, S]
    per_layer_ops += H * 2 * S * S * d_h
    per_layer_act_read += H * (S * d_h * _FP16_BYTES) * 2  # Q + K

    # AV attention: [H, S, S] × [H, S, d_h]
    per_layer_ops += H * 2 * S * S * d_h
    per_layer_act_read += H * (S * S * _FP16_BYTES + S * d_h * _FP16_BYTES) * 2

    # Output projection: [S, H × d_h] × [H × d_h, d_model]
    M, N, K = S, d, H * d_h
    per_layer_ops += 2 * M * N * K
    per_layer_weight += (K * N) * _FP16_BYTES
    per_layer_act_read += (M * K) * _FP16_BYTES
    per_layer_output += (M * N) * _FP16_BYTES

    # FFN
    if spec.ffn_type in ("swiglu", "glu"):
        # gate_proj + up_proj: 2 × [S, d_model] × [d_model, d_ff]
        for _ in range(2):
            M, N, K = S, d_ff, d
            per_layer_ops += 2 * M * N * K
            per_layer_weight += (K * N) * _FP16_BYTES
            per_layer_act_read += (M * K) * _FP16_BYTES
        # down_proj: [S, d_ff] × [d_ff, d_model]
        M, N, K = S, d, d_ff
        per_layer_ops += 2 * M * N * K
        per_layer_weight += (K * N) * _FP16_BYTES
        per_layer_act_read += (M * K) * _FP16_BYTES
        per_layer_output += (M * N) * _FP16_BYTES
    else:
        # FFN L1: [S, d_model] × [d_model, d_ff]
        M, N, K = S, d_ff, d
        per_layer_ops += 2 * M * N * K
        per_layer_weight += (K * N) * _FP16_BYTES
        per_layer_act_read += (M * K) * _FP16_BYTES
        # FFN L2: [S, d_ff] × [d_ff, d_model]
        M, N, K = S, d, d_ff
        per_layer_ops += 2 * M * N * K
        per_layer_weight += (K * N) * _FP16_BYTES
        per_layer_act_read += (M * K) * _FP16_BYTES
        per_layer_output += (M * N) * _FP16_BYTES

    total_ops = per_layer_ops * L
    # LM head / embedding projection at output: [S, d] × [d, vocab]
    total_ops += 2 * S * d * spec.vocab
    weight_bytes = (spec.vocab * d * _FP16_BYTES) + per_layer_weight * L
    act_read_bytes = per_layer_act_read * L
    output_bytes = per_layer_output * L

    return total_ops, act_read_bytes, weight_bytes, output_bytes


def compute_vector_ops(spec: ModelSpec) -> Tuple[float, float]:
    """Compute total vector ops + intermediate UB↔L1 bytes per inference (B=1).

    Returns:
        (vector_flops, intermediate_bytes)
    """
    L = spec.layers
    S = spec.S
    d = spec.d_model
    d_ff = spec.d_ff

    # RMSNorm × 2 per layer: 2 ops/element × S × d × 2 ops (normalize + scale)
    vec_ops = L * 2 * S * d * 2

    # Activation function over FFN intermediate
    if spec.ffn_type in ("swiglu", "glu"):
        vec_ops += L * S * d_ff   # SiLU on gate output

    # Softmax per attention head: H × S × 3 ops/element (exp + sum + div)
    vec_ops += L * spec.n_heads * S * 3

    # Intermediate UB↔L1 traffic: 2× RMSNorm bytes + 2× FFN intermediate bytes
    inter_bytes = L * (2 * S * d * _FP16_BYTES + 2 * S * d_ff * _FP16_BYTES)

    return vec_ops, inter_bytes


def estimate_n_kernels(spec: ModelSpec, *, apply_correction: bool = True) -> int:
    """Estimate number of kernels per inference (each GEMM + each vec op = 1 kernel).

    The naive count ``(n_gemm + n_vec) × layers`` matches the conceptual op count
    but undercounts the **actual** CANN kernel launches by 2.5×–32×, since each
    logical GEMM is split into many tile-level launches at runtime.

    With ``apply_correction=True`` (default), an empirical archetype multiplier
    is applied — calibrated against the 9 measured msprof configs (Issue #2 P1).

    Args:
        spec: ModelSpec
        apply_correction: if False, return the naive uncalibrated count (useful
            for unit tests / introspection).
    """
    n_gemm_per_layer = 7 if spec.arch == "decoder" else 8  # +1 if encoder MLM head
    n_vec_per_layer = 3   # 2× RMSNorm + 1 activation
    base = (n_gemm_per_layer + n_vec_per_layer) * spec.layers

    if not apply_correction:
        return base

    # Archetype correction (P1 Issue #2). Calibration data:
    #   BERT-S128-b1 (small):       base=132, meas=338  → ratio 2.56
    #   Qwen3-prefill-S256-b1 (lg): base=280, meas=7224 → ratio 25.8
    #   Qwen3-prefill-S512-b4 (lg): base=280, meas=9030 → ratio 32.3
    #   Qwen3-decode-Min4 (decode): base=280, meas=1307 → ratio 4.67
    weight_mb_proxy = spec.layers * (4 * spec.d_model ** 2 + 3 * spec.d_model * spec.d_ff) * 2 / 1e6
    if weight_mb_proxy < 600:
        multiplier = 2.5     # small models (BERT/GPT-2 class)
    elif spec.S == 1:
        multiplier = 4.7     # decode regime
    else:
        multiplier = 28.0    # large prefill (Qwen3 class)
    return int(base * multiplier)


def estimate_n_kernels_v5(spec: ModelSpec,
                          params: "Mapping[str, float] | None" = None) -> int:
    """v5 n_kernels: saturating multiplier (replaces v4 28× cliff).

    Calibrated against 4 OOS configs: actual mult on Llama/Qwen2.5/SmolLM2/
    ModernBERT is ~6×, not 28×. v4 28× over-fit Qwen3 prefill.

    Free params:
      nk_mult_base   (default 2.5): small-model multiplier
      nk_mult_max    (default 6.0): asymptotic large-model multiplier
      nk_W_sat       (default 300.0): w_proxy MB at half-saturation
      nk_mult_decode (default 4.7): decode-regime multiplier
    """
    from .physics_v5 import n_kernels_mult_v5
    params = params or {}
    n_gemm_per_layer = 7 if spec.arch == "decoder" else 8
    n_vec_per_layer = 3
    base = (n_gemm_per_layer + n_vec_per_layer) * spec.layers
    weight_mb_proxy = spec.layers * (4 * spec.d_model ** 2 + 3 * spec.d_model * spec.d_ff) * 2 / 1e6
    multiplier = n_kernels_mult_v5(weight_mb_proxy, spec.S, params)
    return int(base * multiplier)


def estimate_n_vector_kernels(spec: ModelSpec) -> int:
    """Estimate number of Vector-unit kernels per inference.

    Each layer has ~3 vector ops (2× RMSNorm/LayerNorm + 1 activation).
    Plus embedding Norm + final Norm + optional LM-head softmax.
    CANN may fuse some consecutive vector ops for small models.

    The count is more stable than ``estimate_n_kernels()`` because CANN's
    tile-splitting primarily affects GEMM kernels, not vector kernels.
    """
    per_layer = 3   # 2× Norm + 1 GeLU/SiLU
    fixed = 2       # input norm + output norm
    if spec.vocab > 10000:
        fixed += 1  # LM head softmax

    base = per_layer * spec.layers + fixed

    # CANN fusion heuristic: small models get more fusion (fewer actual launches)
    weight_mb = spec.layers * (4 * spec.d_model ** 2
                               + 3 * spec.d_model * spec.d_ff) * 2 / 1e6
    if weight_mb < 600:
        return max(1, int(base * 0.8))    # small: CANN fuses some vec ops
    if spec.S == 1:
        return max(1, int(base * 1.5))    # decode: extra KV-cache management kernels
    return max(1, int(base * 2.0))        # large prefill: tiling splits vec kernels


# ─────────────────────────────────────────────────────────────────────────
# KNOWN_MODELS — baseline registry for leave-one-out CV.
# These configs MUST appear as keys in
# ``data/calibration/pipe_baseline_per_model.json`` for the fit step.
# ─────────────────────────────────────────────────────────────────────────
KNOWN_MODELS: Dict[str, ModelSpec] = {
    "BERT-base-S128-b1": ModelSpec(
        name="BERT-base", arch="encoder", layers=12, S=128,
        d_model=768, d_ff=3072, n_heads=12, n_kv_heads=0, d_head=64,
        vocab=30522, ffn_type="standard",
    ),
    "GPT-2-S512-b1": ModelSpec(
        name="GPT-2-small", arch="decoder", layers=12, S=512,
        d_model=768, d_ff=3072, n_heads=12, n_kv_heads=0, d_head=64,
        vocab=50257, ffn_type="standard",
    ),
    "Qwen3-prefill-S512-b4": ModelSpec(
        name="Qwen3-0.6B", arch="decoder", layers=28, S=512,
        d_model=1024, d_ff=3072, n_heads=16, n_kv_heads=8, d_head=128,
        vocab=151936, ffn_type="swiglu",
    ),
    "Qwen3-prefill-S256-b1": ModelSpec(
        name="Qwen3-0.6B", arch="decoder", layers=28, S=256,
        d_model=1024, d_ff=3072, n_heads=16, n_kv_heads=8, d_head=128,
        vocab=151936, ffn_type="swiglu",
    ),
    "Qwen3-decode-Min4-Skv128-b1": ModelSpec(
        name="Qwen3-0.6B-decode", arch="decoder", layers=28, S=1,
        d_model=1024, d_ff=3072, n_heads=16, n_kv_heads=8, d_head=128,
        vocab=151936, ffn_type="swiglu",
    ),
    "Net-Transformer-S256-L1-b1": ModelSpec(
        name="Net-Transformer", arch="encoder", layers=1, S=256,
        d_model=384, d_ff=1536, n_heads=6, n_kv_heads=0, d_head=64,
        vocab=1024,                  # 1024-class classifier head, NOT 10000-vocab embed
        ffn_type="standard",
    ),
}
