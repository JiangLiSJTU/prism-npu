# ET-BERT 在昇腾 910B 上的 FP8/FP4 架构分析报告

> **一句话**：ET-BERT 在 910B 上是 **host 调度 + 片上 L1(mte2) 受限**，不是算力或 HBM 受限；
> 因此 FP8→FP4 的 **4× 算力与 2× 带宽几乎全被浪费**，降精度（FP16→INT8）实测反而**慢 26%**；
> 只有"下一代 fixpipe 吸收量化税"能兑现约 **20%** 收益。结论全部由 910B4 真机实测支撑，PRISM 仅用于 FP8/FP4 外推。

**实验设置**：910B4（CANN+ais_bench+msprof），ET-BERT=BERT-base 拓扑+vocab256（L12 d768 S128）。
FP16/INT8 真机实测（INT8 经 AMCT PTQ），FP8/FP4 用 `prism-predict-pipe --precision` 外推。
数据：`data/calibration/etbert_measured.json`；详证：`docs/findings/etbert_fp8_fp4_910b_analysis.md`。

---

## Q1 — FP8/FP4 混合精度 MFU 怎么公平计算？

**结论**：峰值随精度变，混合精度**无单一峰值**；公平做法 = **按时间加权、各算子相对各自精度峰值归一**，
`MFU = Σ_op (t_op/T)·util_op`（= msprof `cube_util_fp16 + cube_util_int8`）。跨精度别直接比 MFU 数字
（分母峰值变了），应比**有效 TOPS/时延**或拆 `η=η_pipeline·η_tile·η_mem`。

**实测**：ET-BERT cube 利用率 FP16 32→75%（b1→b16）、INT8 14→46%；`int8_ratio` 首次非 0。

**原因**：INT8 cube 峰值 2×、同样 MAC 量半时间干完，但 wall 被 host/mte2 钉住 → cube 更早空闲 → MFU 反而更低。

---

## Q2 — 800T FP4 与 200T FP8 访存带宽需求一样吗？

**结论**：**不一样**。要分两个口径——
- **搬运字节量**：FP4 = ½ × FP8（位宽减半）。
- **维持峰值所需带宽(GB/s)**：`BW需求 ∝ 峰值算力 × 位宽` → FP8→FP4 = **4× × ½ = 2×（翻倍）**。
- 对照 910B FP16→INT8（算力2×/密度2×匹配）：带宽需求**不变**。

**实测/仿真**：维持 compute-bound 的最小带宽——FP8 74 GB/s → FP4 **147 GB/s**（正好 2×）；910B FP16/INT8 同为 232 GB/s。

**原因**：算力放大倍数 vs 位宽缩小倍数是否相等，决定带宽需求是否变化；FP4 算力(4×)跑赢密度(2×)，缺口翻倍。

---

## Q3 — FP8→FP4，MFU 会下降、需更大带宽吗？

**结论**：**会下降、需更大带宽**——但主导机理是 **roofline 左移（计算时间缩得比访存快→CUBE 被饿）**，
而非"tile 变大"（ET-BERT 归约维全 64 倍数，`η_tile=1`，tile 损失≈0）。

**实测**：同 batch INT8 cube 利用率 < FP16（b16 45.5% vs 74.5%、b1 14.3% vs 32.2%）；
mac 时间 INT8≈½FP16（实测 2× 吞吐）；**mte2(访存) 在 FP16/INT8 所有 batch 恒为主导 pipe**。
**🔥 反直觉杀手锏：INT8 端到端比 FP16 更慢**（b1 2.21 vs 1.76 ms）。

**原因**：ET-BERT 非算力受限，砍 mac 不缩 wall；量化引入 requant（fixpipe/AscendQuant，占 INT8 wall 19–35%）反而加时延。
→ 降精度在 host/访存受限模型上**不提速、反更差**——必须先拆调度/访存墙才能兑现低精度。

---

## Q4 — 16MB Cache 兜得住模型吗？加 Cache 降带宽？LLC 命中率/DDR 利用率？精度变化？对齐？

**结论**：16MB 装不下整模型（实测 OM **FP16 165 / INT8 85 MB**），但**不需要**——逐层流式，关键是**单层工作集**
（FP16 14.16 / INT8 7.08 / FP4 3.54 MB）。稳态下 **96MB L2 已让 DDR 近乎空闲**；**精度↓→权重小→L2 命中↑→带宽更省**。

**实测**：
| | LLC/L2 读命中率 | DDR/HBM 利用率 | 访存带宽层级 |
|---|---|---|---|
| FP16 | 74–96% | **0.4%（近空闲）** | HBM 1.4 → L2 2.7 → **L1 122 GB/s** |
| INT8 | **95–99.8%** | 0.4% | （L1 复用吸收, DDR 几乎不动） |

**原因**：① 权重(165MB)>L2(96MB) 但**跨调用复用 + L2 命中**吸收了几乎全部权重流量 → DDR 仅 0.4%，
瓶颈是片上 mte2(L1 填充)+host，**不是 DDR**；② INT8 权重(85MB)更易驻 96MB L2 → 命中更高；
③ **对齐**：910B fractal 按 512B 对齐，低精度加深内维（FP16 16×16 / INT8 16×32 / FP4 16×64）保持 512B，
ET-BERT 各维 64 倍数 → **FP4 下仍零浪费**。
> ⚠️ **实测修正 PRISM**：PRISM 假设"权重>L2 即每次从 HBM 流式重载"→ 高估稳态 HBM 流量；真机 DDR 仅 0.4%。

---

## FP8/FP4 端到端投影（理想 fixpipe + 2× 带宽，锚定实测）

`prism-predict-pipe --precision fp4 --ideal-fixpipe --arch ...2xbw.yaml`（量化税/命中率已反馈进精度档）。

| 场景 | 时延 ms | vs FP16 | 说明 |
|---|---|---|---|
| FP16 实测 | 1.76 | 1.00× | 基线 |
| INT8 实测（含量化税） | 2.21 | **1.26×** | requant 反而更慢 |
| INT8 理想 fixpipe（去 quant+dequant） | 1.43 | 0.82× | 快 18% |
| **FP4 理想 fixpipe** | **1.40** | **0.80×** | ≈INT8 理想，快 ~20% |
| **FP4 理想 fixpipe + 2× 带宽** | **1.40** | 0.79× | **2× 带宽贡献 ~0** |

**投影结论**：① 专门的低精度 fixpipe 电路能把 INT8/FP4 从"负收益"救回 **+20%**；
② 但 **4× 算力与 2× 带宽几乎全浪费**（ET-BERT 受 host/L1 约束、DDR 0.4% 空闲）；
③ 要吃满低精度算力，fixpipe 优化须叠加 **拆 host 调度墙（kernel 融合/大 batch）+ 提 L1/片上带宽**，三者缺一不可。

---

## 总结论（一图流）

```
ET-BERT@910B 瓶颈链：  host 调度墙  ≫  片上 mte2(L1填充)  >  量化税(INT8)  ≫  HBM/DDR(0.4%空闲)
低精度收益被谁吃掉：    ↑没拆           ↑没拆               ↑下一代fixpipe可去   ↑根本不是瓶颈
能兑现的：             FP4 理想 ≈ +20%（仅靠去量化税）；4×算力/2×带宽 ≈ 0 收益
要全兑现：             去量化税(fixpipe) + 拆host(融合/大batch) + 提L1带宽  —— 三者并举
```

**工具与数据**：`models/regime/et_bert.yaml`、`prism-predict-pipe --precision {fp16,int8,fp8,fp4} [--ideal-fixpipe] [--mac-mult]`、
`arch/ascend_910b4_2xbw.yaml`、`data/calibration/etbert_measured.json`、采集脚本 `benchmark/run_etbert_{int8,llc_ddr}_msprof.sh`。
