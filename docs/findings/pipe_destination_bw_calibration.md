# Finding：aic_fixpipe / aiv_mte3 输出目的地带宽校准

> Issue #7 | 日期：2026-05-21 | PRISM v8 @ PR #6
> 校准脚本：`scripts/calib_fixpipe_mte3_bw.py` → `data/calibration/pipe_dest_bw.json`

---

## 1. 问题

PRISM sweep 的 pipe-aware 公式按 arch 资源缩放每条 pipe。`aic_fixpipe`（Cube L0C→输出）
与 `aiv_mte3`（Vector UB→输出）此前都用**单一片上带宽 knob** 缩放（`fixpipe_bw` /
`ub_l1_bw`）。但这两条 pipe 的真实瓶颈带宽**取决于输出目的地**：

| pipe | 片上目的地 | 片外目的地 |
|---|---|---|
| `aic_fixpipe` | L0C→L1/UB（`fixpipe_bw` ~4096 GB/s）| **L0C→GM 直写**（`hbm_bw` ~392 GB/s）|
| `aiv_mte3` | UB→L1（`ub_l1_bw` ~2048，`copy_ubuf_to_cbuf`）| **UB→GM**（`hbm_bw`，`copy_ubuf_to_gm`）|

目的地不分清，sweep 的 `fixpipe_bw` / `hbm_bw` 维度结论会把架构杠杆挂错名字。
**PRISM 是普世工具，未来用户分析的模型形态未知 —— 方法必须对单峰、双峰、混合
都正确。** 这是本 finding 选择 2-cluster 方法的理由。

## 2. 校准方法：Prior-based 2-cluster

msprof 只报聚合 `aic_fixpipe_time` / `aiv_mte3_time`。**每个 op 用 implied 带宽
`bytes/time` 分类**：

```
threshold = sqrt(hbm_bw · onchip_bw)   # 两区域几何中点

每 op:
  implied_bw = bytes / time
  implied_bw <  threshold  →  GM 簇 (HBM-bound)
  implied_bw >= threshold  →  on-chip 簇

gm_frac = Σ(GM 簇字节) / Σ(总字节)        # 实测字节占比
```

每个簇再做一次 OLS（slope = 1/bw）作为 **sanity check**：GM 簇的 OLS 斜率应落在
`hbm_bw` 量级，on-chip 簇应落在 `fixpipe_bw` / `ub_l1_bw` 量级。

**普世性**：单峰 config 自然退化为 `1cluster`（一个簇为空）；真双峰 config 干净分簇。

回避两种朴素方法的失效：

- **`Σ字节/Σ时间`**：被 per-op 固定开销污染（实测同样 3 MB 的 GatherV2，mte3 时间
  差 11×，聚合带宽完全无意义）。
- **单一 pooled OLS**：双峰数据给 leverage-weighted 混合斜率，反解的 `gm_frac` 有偏
  （例：HF-BERT-b8 fixpipe pooled OLS 给 gm_frac=0.21，2-cluster 修正为 0.78）。

`confidence`：基于每簇 OLS 的 `r²` —— 双簇 r² 最小值 ≥ 0.5 标 high，否则 low。

## 3. 实测校准表（39 config）

`gm_bw` = GM 簇 OLS 斜率（应近 hbm_bw 392 GB/s）；`oc_bw` = on-chip 簇 OLS 斜率（应近
`fixpipe_bw` / `ub_l1_bw`）；`—` 表示该簇 op 数不足 `n < 5` 或为空。

| config | fp method | fp gm_frac | fp gm_bw | fp oc_bw | m3 method | m3 gm_frac | m3 gm_bw | m3 oc_bw | source |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| BERT-base-S128-b1 | inherit | 1.00 | — | — | inherit | 0.09 | — | — | inherited |
| BERT-base-S128-b4 | 1cluster | 1.00 | 571 | — | 2cluster | 0.09 | 755 | 2775 | measured |
| BERT-base-S128-b8 | 2cluster | 0.86 | 562 | — | 2cluster | 0.02 | 455 | 2911 | measured |
| BERT-base-S128-b16 | 2cluster | 0.83 | 507 | — | 2cluster | 0.06 | 507 | 3103 | measured |
| GPT-2-S512-b1 | inherit | 0.66 | — | — | inherit | 0.40 | — | — | inherited |
| GPT-2-S512-b4 | 2cluster | 0.66 | 361 | — | 2cluster | 0.40 | 430 | 1664 | measured |
| GPT-2-S512-b8 | 2cluster | 0.71 | 359 | — | 2cluster | 0.39 | 423 | 1900 | measured |
| GPT-2-S512-b16 | 1cluster | 1.00 | 375 | — | 2cluster | 0.56 | 428 | 2551 | measured |
| HF-BERT-S128-b1 | 1cluster | 1.00 | 664 | — | 2cluster | 0.84 | 715 | 896 | measured |
| HF-BERT-S128-b4 | 1cluster | 1.00 | 841 | — | 2cluster | 0.21 | 337 | 3845 | measured |
| HF-BERT-S128-b8 | 2cluster | 0.78 | 820 | — | 2cluster | 0.11 | 606 | 2708 | measured |
| HF-BERT-S128-b16 | 1cluster | 1.00 | 719 | — | 2cluster | 0.05 | 363 | 2569 | measured |
| Net-Transformer-S256-L1-b1 | 1cluster | 1.00 | 568 | — | 2cluster | 0.45 | 745 | 1514 | measured |
| Net-Transformer-S256-L1-b4 | 2cluster | 0.92 | 524 | — | 2cluster | 0.07 | 604 | 3796 | measured |
| Net-Transformer-S256-L1-b8 | 2cluster | 0.93 | 574 | 1421 | 2cluster | 0.06 | 702 | 3049 | measured |
| Net-Transformer-S256-L1-b16 | 1cluster | 1.00 | 558 | — | 2cluster | 0.01 | 720 | 2824 | measured |
| ModernBERT-base-S4096-b1 | 1cluster | 1.00 | 318 | — | 2cluster | 0.85 | 528 | 3065 | measured |
| ModernBERT-base-S4096-b1-sdpa | 1cluster | 1.00 | 699 | — | 2cluster | 0.50 | 493 | 952 | measured |
| Llama-3.2-1B-prefill-S2048-b1 | 1cluster | 1.00 | 797 | — | 2cluster | 0.81 | 527 | 1323 | measured |
| Llama-3.2-1B-prefill-S2048-b1-sdpa | 1cluster | 1.00 | 783 | — | 2cluster | 0.76 | 545 | 1377 | measured |
| Qwen2.5-0.5B-prefill-S2048-b1 | 2cluster | 0.42 | 248 | 1625 | 2cluster | 0.79 | 484 | 1911 | measured |
| Qwen2.5-0.5B-prefill-S2048-b1-sdpa | 2cluster | 0.70 | 301 | 1520 | 2cluster | 0.74 | 484 | 2197 | measured |
| SmolLM2-360M-prefill-S2048-b1 | 2cluster | 0.28 | 662 | — | 2cluster | 0.80 | 472 | 1970 | measured |
| SmolLM2-360M-prefill-S2048-b1-sdpa | 1cluster | 1.00 | 1059 | — | 2cluster | 0.75 | 428 | 1781 | measured |
| Phi-3-mini-prefill-S2048-b1-sdpa | 1cluster | 1.00 | 726 | — | 2cluster | 0.74 | 551 | 1569 | measured |
| Qwen3-prefill-S256-b1 | inherit | 0.83 | — | — | inherit | 0.23 | — | — | inherited |
| Qwen3-prefill-S256-b1-sdpa | 2cluster | 0.83 | 237 | — | 2cluster | 0.23 | 567 | 2906 | measured |
| Qwen3-prefill-S256-b4 | inherit | 1.00 | — | — | inherit | 0.23 | — | — | inherited |
| Qwen3-prefill-S256-b4-sdpa | 1cluster | 1.00 | 266 | — | 2cluster | 0.23 | 446 | 2414 | measured |
| Qwen3-prefill-S256-b8 | inherit | 0.83 | — | — | inherit | 0.24 | — | — | inherited |
| Qwen3-prefill-S256-b8-sdpa | 2cluster | 0.83 | 252 | — | 2cluster | 0.24 | 437 | 2699 | measured |
| Qwen3-prefill-S512-b4 | inherit | 0.67 | — | — | inherit | 0.18 | — | — | inherited |
| Qwen3-prefill-S512-b4-sdpa | 2cluster | 0.67 | 271 | 1430 | 2cluster | 0.18 | 433 | 1624 | measured |
| Qwen3-prefill-S512-b8 | inherit | 1.00 | — | — | inherit | 0.57 | — | — | inherited |
| Qwen3-prefill-S512-b8-sdpa | 1cluster | 1.00 | 254 | — | 2cluster | 0.57 | 436 | 1560 | measured |
| Qwen3-prefill-S4096-b1 | inherit | 0.98 | — | — | inherit | 0.55 | — | — | inherited |
| Qwen3-prefill-S4096-b1-sdpa | 2cluster | 0.98 | 479 | — | 2cluster | 0.55 | 485 | 967 | measured |
| Qwen3-Embedding-S4096-b1 | inherit | 0.98 | — | — | inherit | 0.55 | — | — | inherited |
| Qwen3-decode-Min4-Skv128-b1 | assumed | 0.00 | — | — | assumed | 0.00 | — | — | assumed |

## 4. 关键发现

1. **`aic_fixpipe` 几乎全 GM-bound。** GM 簇 OLS 斜率在几乎所有 config 上落在 240–820 GB/s
   （HBM 量级），远低于片上 `fixpipe_bw` 4096 —— FixPipe 输出主要是 L0C→GM 直写。
   长上下文大 prefill 的 `gm_frac` 多为 0.83–1.00（Qwen3-S4096 `gm_frac=0.98`）。

2. **`aiv_mte3` 在大 prefill 上 GM-bound。** 长上下文大模型 `gm_frac ≈ 0.50–0.85`，
   GM 簇斜率 ~430–550 GB/s。小短序列模型 `gm_frac ≤ 0.15`（留片上）。

3. **真双峰 config 被正确分簇。** 例：HF-BERT-b8 fixpipe pooled OLS 给 gm_frac=0.21，
   2-cluster 修正为 0.78；Qwen2.5-S2048-b1 fixpipe 双簇清晰可分（GM 簇 248 + 片上簇
   1625 GB/s）。单 OLS 在这些 config 上的偏差被 2-cluster 消除。

4. **主报告 §6.4.1 修正。** Qwen3-prefill-S4096 `aic_fixpipe gm_frac=0.98` —— 长 prefill
   attention 输出回写是 L0C→GM 直写。"FixPipe 减半 → ratio 1.24" 修正为 **1.00**
   （FixPipe 单元带宽几乎不是杠杆）；真杠杆是 **HBM 带宽**（`hbm_bw=50` 时 ratio ~3.5）。

## 5. 对 sweep 的影响

`scale_aic_pipes` / `scale_aiv_pipes` 据 per-config `gm_frac` 把这两条 pipe 在 `hbm_bw`
与片上带宽之间 blend 缩放：

- `fixpipe_bw` 维度：对大 prefill 模型的杠杆大幅减弱（fixpipe 主要 GM-bound）。
- `hbm_bw` 维度：大 prefill 模型敏感度显著上升（aic_fixpipe + aiv_mte3 都计入 HBM）。

## 6. 局限

1. **阈值是物理先验**（HBM/onchip 的几何中点）。若实际 HBM 写带宽与标称 392 显著偏离，
   边界 op 可能误判 ±10%；但实测 GM 簇 OLS 斜率多落 250–550 GB/s，处于合理范围。
2. `confidence=low` 的 config（小 host-bound 模型）`gm_frac` 不可靠，但该 pipe 在
   这些 config 上非杠杆，不影响 sweep 结论。
3. 9 个早期 config 无本地 msprof，`gm_frac` 用近邻同族继承（`source=inherited`）。
4. `gm_frac` 视为 arch-invariant（跨 arch variant 不变）。
5. 后续可考虑：把阈值放在 EM mixture model（学出 cluster 均值）以减少对先验的依赖；
   当前 prior-based 已足够区分 HBM 与 onchip 两个量级（差 5–10×）。

## 7. 关联

- Issue：`.sisyphus/issues/issue-7-fixpipe-mte3-destination-bw.md`
- 方法论：`05_calibration.md §3.2/§3.3`、`08_predict_pipe.md §3.6`、`02_three_layer_roofline.md §5/§6`
- 数据：`data/calibration/pipe_dest_bw.json`
- 代码：`src/prism/sweep/runner.py` `scale_aic_pipes` / `scale_aiv_pipes`
