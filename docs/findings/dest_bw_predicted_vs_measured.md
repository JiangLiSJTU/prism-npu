# Predicted-vs-Measured: aic_fixpipe / aiv_mte3 destination bandwidth

> Issue #7 follow-up validation. Quantifies modeling error reduction from
> the OLD model (constant on-chip bandwidth: fixpipe_bw=4096 / ub_l1_bw=2048 GB/s)
> vs the NEW model (per-config HBM↔on-chip blend by `gm_frac`),
> against per-config OLS-fit empirical bandwidth from msprof.

- HBM bandwidth: **392 GB/s**
- On-chip ref (fixpipe / ub_l1): **4096 / 2048 GB/s**
- Configs in Part A: 51 (msprof-measured, confidence=high)
- Configs in Part B: 25 (all with measured `wall_clock_us`)

## Part A — per-pipe effective bandwidth, OLD vs NEW

Each row: one (model family, pipe) pair. `err` = (predicted − measured) / measured. Positive err means the model **overestimates** the available bandwidth (underestimates pipe time → underestimates HBM sensitivity).

### aic_fixpipe (L0C → output)

| Family | n | OLD median err | OLD range | NEW median err | NEW range |
|---|---:|---:|---|---:|---|
| BERT-base | 3 | +812.7% | [+787.3, +944.9] | +0.0% | [+0.0, +0.0] |
| GPT-2 | 3 | +674.0% | [+622.5, +944.9] | +0.0% | [-0.0, -0.0] |
| HF-BERT | 2 | +840.9% | [+736.8, +944.9] | +0.0% | [-0.0, -0.0] |
| Llama-3.2 | 2 | +944.9% | [+944.9, +944.9] | +0.0% | [+0.0, +0.0] |
| ModernBERT | 2 | +944.9% | [+944.9, +944.9] | +0.0% | [+0.0, +0.0] |
| Net-Trans | 4 | +913.5% | [+864.2, +944.9] | +0.0% | [-0.1, +0.0] |
| Phi-3 | 1 | +944.9% | [+944.9, +944.9] | +0.0% | [+0.0, +0.0] |
| Qwen2.5 | 2 | +530.4% | [+399.6, +661.1] | -0.0% | [-0.0, -0.0] |
| Qwen3-prefill | 6 | +852.5% | [+633.5, +944.9] | +0.0% | [-0.1, -0.0] |
| SmolLM2 | 2 | +606.3% | [+267.7, +944.9] | +0.1% | [+0.0, +0.1] |

### aiv_mte3 (UB → output)

| Family | n | OLD median err | OLD range | NEW median err | NEW range |
|---|---:|---:|---|---:|---|
| BERT-base | 1 | +8.3% | [+8.3, +8.3] | -0.1% | [-0.1, -0.1] |
| GPT-2 | 3 | +167.2% | [+162.6, +234.5] | -0.0% | [-0.0, -0.0] |
| HF-BERT | 3 | +48.1% | [+22.2, +355.7] | +0.0% | [-0.1, +0.0] |
| Llama-3.2 | 2 | +330.8% | [+319.5, +342.0] | +0.0% | [-0.0, -0.0] |
| ModernBERT | 2 | +284.9% | [+210.6, +359.2] | -0.1% | [-0.1, +0.0] |
| Net-Trans | 3 | +23.9% | [+4.5, +190.4] | -0.0% | [-0.1, +0.2] |
| Phi-3 | 1 | +312.4% | [+312.4, +312.4] | -0.1% | [-0.1, -0.1] |
| Qwen2.5 | 2 | +322.9% | [+311.4, +334.3] | +0.0% | [+0.0, +0.0] |
| Qwen3-prefill | 5 | +100.5% | [+76.7, +241.0] | -0.0% | [-0.1, -0.0] |
| SmolLM2 | 2 | +327.4% | [+316.1, +338.8] | +0.0% | [+0.0, +0.0] |

**Reading**: the OLD model is the pre-#7 sweep formula —
`pipe_time_new = pipe_time_baseline × (onchip_baseline / onchip_variant)`. Under this rule, the *effective* bandwidth ridden by the pipe was implicitly
the constant on-chip reference (4096 / 2048 GB/s). The empirical OLS slope
shows actual effective bandwidth is 5–10× lower on most large prefill configs
— because the store goes mostly to GM (HBM-bound). The NEW model fixes this
via per-config `gm_frac` blend (err ≈ 0% by construction, modulo OLS noise).

## Part B — wall-clock baseline reproduction (sanity)

End-to-end `predict_wallclock_v3` at variant=baseline, full Issue #7 code path
(gm_frac injection + blend scaling). At baseline arch the blend factor is
**1.0 by construction**, so this confirms no off-by-one bugs from the new code.

| Family | n | abs(err)% median | range |
|---|---:|---:|---|
| BERT-base | 1 | 0.01% | [0.01, 0.01] |
| GPT-2 | 1 | 0.00% | [0.00, 0.00] |
| Llama-3.2 | 2 | 0.00% | [0.00, 0.00] |
| ModernBERT | 2 | 0.00% | [0.00, 0.00] |
| Phi-3 | 1 | 0.00% | [0.00, 0.00] |
| Qwen2.5 | 2 | 0.00% | [0.00, 0.00] |
| Qwen3-Embedding | 1 | 0.00% | [0.00, 0.00] |
| Qwen3-decode | 1 | 0.06% | [0.06, 0.06] |
| Qwen3-prefill | 12 | 0.00% | [0.00, 0.02] |
| SmolLM2 | 2 | 0.00% | [0.00, 0.00] |

**Overall**: 25 configs, median abs(err) **0.00%**, max **0.06%**. All below the 10% hard gate enforced by `tests/test_sweep.py::test_predict_wallclock_v3_baseline_reproduction`.

## Per-config detail (Part A)

| Config | Pipe | gm_frac | bw_meas (GB/s) | bw_old | err_old | bw_new | err_new |
|---|---|---:|---:|---:|---:|---:|---:|
| BERT-base-S128-b16 | aic_fixpipe | 0.833 | 461.6 | 4096 | +787.3% | 462 | +0.0% |
| BERT-base-S128-b4 | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| BERT-base-S128-b8 | aic_fixpipe | 0.860 | 448.8 | 4096 | +812.7% | 449 | +0.0% |
| GPT-2-S512-b16 | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| GPT-2-S512-b4 | aic_fixpipe | 0.659 | 566.9 | 4096 | +622.5% | 567 | -0.0% |
| GPT-2-S512-b8 | aic_fixpipe | 0.713 | 529.2 | 4096 | +674.0% | 529 | +0.0% |
| HF-BERT-S128-b16 | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| HF-BERT-S128-b8 | aic_fixpipe | 0.780 | 489.5 | 4096 | +736.8% | 489 | -0.0% |
| Llama-3.2-1B-prefill-S2048-b1 | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| Llama-3.2-1B-prefill-S2048-b1-sdpa | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| ModernBERT-base-S4096-b1 | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| ModernBERT-base-S4096-b1-sdpa | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| Net-Transformer-S256-L1-b1 | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| Net-Transformer-S256-L1-b16 | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| Net-Transformer-S256-L1-b4 | aic_fixpipe | 0.915 | 424.8 | 4096 | +864.2% | 425 | -0.0% |
| Net-Transformer-S256-L1-b8 | aic_fixpipe | 0.934 | 417.1 | 4096 | +882.0% | 417 | -0.1% |
| Phi-3-mini-prefill-S2048-b1-sdpa | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| Qwen2.5-0.5B-prefill-S2048-b1 | aic_fixpipe | 0.423 | 819.9 | 4096 | +399.6% | 820 | -0.0% |
| Qwen2.5-0.5B-prefill-S2048-b1-sdpa | aic_fixpipe | 0.700 | 538.2 | 4096 | +661.1% | 538 | -0.0% |
| Qwen3-prefill-S256-b1-sdpa | aic_fixpipe | 0.828 | 464.0 | 4096 | +782.8% | 464 | +0.0% |
| Qwen3-prefill-S256-b4-sdpa | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| Qwen3-prefill-S256-b8-sdpa | aic_fixpipe | 0.828 | 464.0 | 4096 | +782.8% | 464 | +0.0% |
| Qwen3-prefill-S4096-b1-sdpa | aic_fixpipe | 0.976 | 400.7 | 4096 | +922.2% | 401 | -0.0% |
| Qwen3-prefill-S512-b4-sdpa | aic_fixpipe | 0.671 | 558.4 | 4096 | +633.5% | 558 | -0.1% |
| Qwen3-prefill-S512-b8-sdpa | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| SmolLM2-360M-prefill-S2048-b1 | aic_fixpipe | 0.283 | 1114.1 | 4096 | +267.7% | 1115 | +0.1% |
| SmolLM2-360M-prefill-S2048-b1-sdpa | aic_fixpipe | 1.000 | 392.0 | 4096 | +944.9% | 392 | +0.0% |
| BERT-base-S128-b8 | aiv_mte3 | 0.020 | 1890.5 | 2048 | +8.3% | 1888 | -0.1% |
| GPT-2-S512-b16 | aiv_mte3 | 0.555 | 612.2 | 2048 | +234.5% | 612 | +0.0% |
| GPT-2-S512-b4 | aiv_mte3 | 0.396 | 766.5 | 2048 | +167.2% | 766 | -0.0% |
| GPT-2-S512-b8 | aiv_mte3 | 0.385 | 779.8 | 2048 | +162.6% | 780 | -0.0% |
| HF-BERT-S128-b1 | aiv_mte3 | 0.842 | 449.4 | 2048 | +355.7% | 449 | +0.0% |
| HF-BERT-S128-b16 | aiv_mte3 | 0.053 | 1675.8 | 2048 | +22.2% | 1673 | -0.1% |
| HF-BERT-S128-b8 | aiv_mte3 | 0.114 | 1382.5 | 2048 | +48.1% | 1382 | -0.0% |
| Llama-3.2-1B-prefill-S2048-b1 | aiv_mte3 | 0.810 | 463.3 | 2048 | +342.0% | 463 | -0.0% |
| Llama-3.2-1B-prefill-S2048-b1-sdpa | aiv_mte3 | 0.756 | 488.2 | 2048 | +319.5% | 488 | +0.0% |
| ModernBERT-base-S4096-b1 | aiv_mte3 | 0.850 | 446.0 | 2048 | +359.2% | 446 | +0.0% |
| ModernBERT-base-S4096-b1-sdpa | aiv_mte3 | 0.499 | 659.3 | 2048 | +210.6% | 659 | -0.1% |
| Net-Transformer-S256-L1-b1 | aiv_mte3 | 0.451 | 705.2 | 2048 | +190.4% | 705 | -0.0% |
| Net-Transformer-S256-L1-b16 | aiv_mte3 | 0.011 | 1959.5 | 2048 | +4.5% | 1957 | -0.1% |
| Net-Transformer-S256-L1-b8 | aiv_mte3 | 0.056 | 1653.6 | 2048 | +23.9% | 1656 | +0.2% |
| Phi-3-mini-prefill-S2048-b1-sdpa | aiv_mte3 | 0.740 | 496.6 | 2048 | +312.4% | 496 | -0.1% |
| Qwen2.5-0.5B-prefill-S2048-b1 | aiv_mte3 | 0.791 | 471.6 | 2048 | +334.3% | 472 | +0.0% |
| Qwen2.5-0.5B-prefill-S2048-b1-sdpa | aiv_mte3 | 0.737 | 497.8 | 2048 | +311.4% | 498 | +0.0% |
| Qwen3-prefill-S256-b4-sdpa | aiv_mte3 | 0.228 | 1043.4 | 2048 | +96.3% | 1043 | -0.0% |
| Qwen3-prefill-S256-b8-sdpa | aiv_mte3 | 0.238 | 1021.3 | 2048 | +100.5% | 1021 | -0.0% |
| Qwen3-prefill-S4096-b1-sdpa | aiv_mte3 | 0.547 | 618.8 | 2048 | +231.0% | 619 | -0.0% |
| Qwen3-prefill-S512-b4-sdpa | aiv_mte3 | 0.182 | 1158.8 | 2048 | +76.7% | 1158 | -0.1% |
| Qwen3-prefill-S512-b8-sdpa | aiv_mte3 | 0.571 | 600.5 | 2048 | +241.0% | 600 | -0.0% |
| SmolLM2-360M-prefill-S2048-b1 | aiv_mte3 | 0.802 | 466.7 | 2048 | +338.8% | 467 | +0.0% |
| SmolLM2-360M-prefill-S2048-b1-sdpa | aiv_mte3 | 0.748 | 492.2 | 2048 | +316.1% | 492 | +0.0% |

---

Regenerate: `python3 scripts/validate_dest_bw_predictions.py`

Source data: `data/calibration/pipe_dest_bw.json` (per-config gm_frac),
`data/calibration/pipe_baseline_per_model.json` (msprof measurements).
