"""v7 splits (SDPA-aware, Issue #2 v7).

Mix of:
- non-Qwen3 eager configs (already known to fit fine in v6.1)
- Qwen3-sdpa configs (Qwen3 family under SDPA — no special bucket)

VAL splits are designed to test:
- VAL_size: cross-architecture generalization within AIV_BOUND bucket
  (fit on ModernBERT, predict Llama/Qwen2.5/SmolLM2 — same as v6.1)
- VAL_sdpa_long_S: within-bucket long-S extrapolation (Qwen3-S4096-sdpa)
- VAL_sdpa_batch: batch extrapolation under SDPA (b=8 held out)
"""
from __future__ import annotations
from typing import List, Tuple


# (config_name, model_yaml_path, batch)
TRAIN_CONFIGS_V7: List[Tuple[str, str, int]] = [
    # BALANCED bucket anchors (small/shallow)
    ("BERT-base-S128-b1",            "models/regime/bert_base.yaml",                         1),
    ("GPT-2-S512-b1",                "models/regime/gpt2_small.yaml",                        1),
    # AIC_DECODE anchor
    ("Qwen3-decode-Min4-Skv128-b1",  "models/regime/qwen3_0.6b.yaml",                        1),
    # AIV_BOUND anchor (encoder/GLU)
    ("ModernBERT-base-S4096-b1",     "models/regime/modernbert_base_prefill_S4096.yaml",     1),
    # Qwen3-sdpa anchors (BALANCED via spec rules)
    ("Qwen3-prefill-S256-b1-sdpa",   "models/regime/qwen3_0.6b_prefill_S256.yaml",           1),
    ("Qwen3-prefill-S256-b4-sdpa",   "models/regime/qwen3_0.6b_prefill_S256.yaml",           4),
    ("Qwen3-prefill-S512-b4-sdpa",   "models/regime/qwen3_0.6b_prefill_S512.yaml",           4),
]

# Validation: cross-architecture within AIV_BOUND
VAL_SIZE_V7: List[Tuple[str, str, int]] = [
    ("Llama-3.2-1B-prefill-S2048-b1",       "models/regime/llama_3_2_1b_prefill_S2048.yaml",       1),
    ("Qwen2.5-0.5B-prefill-S2048-b1",       "models/regime/qwen2_5_0_5b_prefill_S2048.yaml",       1),
    ("SmolLM2-360M-prefill-S2048-b1",       "models/regime/smollm2_360m_prefill_S2048.yaml",       1),
]

# Validation: within-bucket long-S extrapolation under SDPA
VAL_SDPA_LONG_S_V7: List[Tuple[str, str, int]] = [
    ("Qwen3-prefill-S4096-b1-sdpa",  "models/regime/qwen3_0.6b_prefill_S4096.yaml",          1),
]

# Validation: batch extrapolation under SDPA
VAL_SDPA_BATCH_V7: List[Tuple[str, str, int]] = [
    ("Qwen3-prefill-S256-b8-sdpa",   "models/regime/qwen3_0.6b_prefill_S256.yaml",           8),
    ("Qwen3-prefill-S512-b8-sdpa",   "models/regime/qwen3_0.6b_prefill_S512.yaml",           8),
]

# Issue #3 Phase 3 — pure-OOS SDPA validation (4 non-Qwen3 families).
# These are NEVER in TRAIN — they test if v8's coefficients transfer when
# BOTH architecture AND attn_impl differ from training anchors.
# Reference comparison: each has a matching eager-path config in baseline
# (without -sdpa suffix), letting us measure SDPA speedup per family.
VAL_SDPA_OOS_V7: List[Tuple[str, str, int]] = [
    ("ModernBERT-base-S4096-b1-sdpa",       "models/regime/modernbert_base_prefill_S4096.yaml",    1),
    ("Llama-3.2-1B-prefill-S2048-b1-sdpa",  "models/regime/llama_3_2_1b_prefill_S2048.yaml",       1),
    ("Qwen2.5-0.5B-prefill-S2048-b1-sdpa",  "models/regime/qwen2_5_0_5b_prefill_S2048.yaml",       1),
    ("SmolLM2-360M-prefill-S2048-b1-sdpa",  "models/regime/smollm2_360m_prefill_S2048.yaml",       1),
]
