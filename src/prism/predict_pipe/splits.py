"""Train / Val / Test splits for v5 fit (Issue #2 v5 with overfit prevention).

Design principle (per user mandate 2026-05-15): v4 over-fit on 6 configs
because all training points fell in w_proxy ∈ [89, 230] MB; Llama at
w_proxy=2147 MB blew up 12×. v5 must:

1. Hold out the LARGE w_proxy configs (Llama, Qwen2.5, SmolLM2-360M) as
   VAL_size — testing extrapolation along the dimension that broke v4.
2. Hold out batch>1 configs as VAL_batch — testing batch axis extrapolation.
3. Hold out one model family per LOMO fold for final cross-val.

The TRAIN set is 13 configs spanning w_proxy ∈ [89, 230] MB (small/mid
models, b=1 only + one Qwen3 b=4 anchor); v5 must extrapolate to VAL_size
without exploding. Target: VAL_size MAE < 30% (vs v4's 137-1156%).
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple


_REPO = Path(__file__).resolve().parents[3]

# (config_name, model_yaml_path, batch)
TRAIN_CONFIGS: List[Tuple[str, str, int]] = [
    ("BERT-base-S128-b1",            "models/regime/bert_base.yaml",                         1),
    ("GPT-2-S512-b1",                "models/regime/gpt2_small.yaml",                        1),
    ("HF-BERT-S128-b1",              "models/regime/hf_bert.yaml",                           1),
    ("Net-Transformer-S256-L1-b1",   "models/regime/net_transformer.yaml",                   1),
    ("Qwen3-prefill-S256-b1",        "models/regime/qwen3_0.6b_prefill_S256.yaml",           1),
    ("Qwen3-prefill-S4096-b1",       "models/regime/qwen3_0.6b_prefill_S4096.yaml",          1),
    ("Qwen3-decode-Min4-Skv128-b1",  "models/regime/qwen3_0.6b.yaml",                        1),
    ("Qwen3-prefill-S512-b4",        "models/regime/qwen3_0.6b_prefill_S512.yaml",           4),
    ("ModernBERT-base-S4096-b1",     "models/regime/modernbert_base_prefill_S4096.yaml",     1),
]

# Batch extrapolation validation (same model families, b=4/8/16)
# NOTE: BERT/GPT-2 batches have host_gap=0 in baseline (measured without step_trace),
# so wall_clock test is degenerate. Kept for reference; fitter skips them.
VAL_BATCH_CONFIGS: List[Tuple[str, str, int]] = [
    ("Qwen3-prefill-S256-b4",        "models/regime/qwen3_0.6b_prefill_S256.yaml",           4),
    ("Qwen3-prefill-S256-b8",        "models/regime/qwen3_0.6b_prefill_S256.yaml",           8),
    ("Qwen3-prefill-S512-b8",        "models/regime/qwen3_0.6b_prefill_S512.yaml",           8),
]

# Size extrapolation validation — the hard test that v4 failed catastrophically.
# Qwen3-S4096-b1 was previously in VAL_size as "long-S anchor outlier"; v6 moved
# it back to TRAIN (per-bucket fit can absorb it without polluting other buckets).
VAL_SIZE_CONFIGS: List[Tuple[str, str, int]] = [
    ("Llama-3.2-1B-prefill-S2048-b1",       "models/regime/llama_3_2_1b_prefill_S2048.yaml",       1),
    ("Qwen2.5-0.5B-prefill-S2048-b1",       "models/regime/qwen2_5_0_5b_prefill_S2048.yaml",       1),
    ("SmolLM2-360M-prefill-S2048-b1",       "models/regime/smollm2_360m_prefill_S2048.yaml",       1),
]


def resolve_path(p: str) -> Path:
    """Resolve a relative path within the repo."""
    abs_path = _REPO / p
    if not abs_path.exists():
        raise FileNotFoundError(f"path not found: {abs_path}")
    return abs_path


def all_splits():
    """Return (train, val_batch, val_size) as resolved tuples."""
    return TRAIN_CONFIGS, VAL_BATCH_CONFIGS, VAL_SIZE_CONFIGS
