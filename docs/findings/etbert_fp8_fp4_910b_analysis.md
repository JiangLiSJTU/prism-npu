# ET-BERT 在昇腾 910B 上的 FP8/FP4 架构分析备忘录

> **载体模型**：ET-BERT（加密流量分类 / AI-DPI，BERT-base 12 层 encoder + 字节级 vocab=256）
> **目标**：用 **910B 实测**（msprof / ais_bench）+ PRISM 仿真，回答下一代芯片 FP8/FP4 的
> 四个架构问题。FP8/FP4 无对应硅，故用 910B 的 **FP16/INT8** 做密度/算力类比，PRISM 负责外推。
> **日期**：2026-06-04 ｜ 关联记忆：`project_prism_fp8_fp4_analysis`

---

## 0. 背景与精度映射

下一代芯片将出现 FP8/FP4 混合精度。910B 无 FP8/FP4 单元，但其 **FP16↔INT8** 可类比 **位宽密度**，
PRISM 再把"算力倍数"外推到问题给定的非对称下一代（FP8=200T / FP4=800T）。

| 维度 | 下一代（问题给定） | 910B 可实测类比 | 比例 |
|---|---|---|---|
| 数据位宽（密度） | FP8=1B → FP4=0.5B | INT8=1B → INT4=0.5B | ÷2 每档 |
| 算力 | FP8=200T → FP4=**800T** | FP16=280T → INT8=560T | 下一代 **×4** / 910B ×2 |

> 关键：下一代 **算力涨 4× 但位宽只缩 2×**；910B FP16→INT8 是 **算力 2× / 位宽 2×（匹配）**。
> 二者是否匹配，决定带宽需求是否变化（见 Q2）。

**ET-BERT 基础量（PRISM `compute_gemm_ops` 实算，B=1）**：22.40 G ops、权重 FP16 **170.3 MB**、
**单层权重工作集 14.16 MB**（INT8 7.08 / FP4 3.54）、激活读 40.1 MB、AIV 中间量 23.6 MB。

---

## 1. 证据分级（先讲清可信度）

| 结论涉及 | 证据等级 | 来源 |
|---|---|---|
| FP16 下 CUBE 利用率、访存受限、host 开销 | ✅ **910B 实测** | msprof / ais_bench（原厂） |
| INT8 跨精度缩放 | ✅ **910B 实测（2026-06-05 补齐）** | AMCT PTQ→ATC→msprof，见 §2.5 |
| LLC 命中率 / DDR 带宽利用率 | ✅ **910B 实测（2026-06-05 补齐）** | msprof L2Cache/Memory+sys-hardware-mem，见 §2.5 |
| FP8/FP4 | ❗ **无硅，PRISM 外推** | `prism-predict-pipe --precision fp8/fp4` |

> ✅ **2026-06-05 实测更新**：910B4 服务器恢复在线，已在真机直接实测 ET-BERT（非 BERT-base 代理）
> FP16+INT8 共 8 配置，三项空白（INT8 / LLC 命中率 / DDR 利用率）**全部补齐**。数据见 §2.5 与
> `data/calibration/etbert_measured.json`。下面 §2.1–2.3 的 BERT-base 实测保留作历史对照。

> ⚠️ 仓库实测的是 **BERT-base（12 层，vocab=30522）**。ET-BERT 的 12 层 encoder 与之**逐字节相同**，
> 只有 embedding/分类头不同（vocab 256 vs 30522）。逐层 CUBE 利用率、pipe 构成、host gap **1:1 适用**；
> ET-BERT 因 vocab 极小，更偏算力/host 受限、更不偏 embedding 访存。

---

## 2. 910B 实测数据（FP16，原厂 msprof / ais_bench）

### 2.1 CUBE 利用率（`cube_util_extracted.json`，msprof ArithmeticUtilization）

| 配置 | n_ops | total_aicore_time | **cube_util_fp16** | int8 | avg_block_dim |
|---|---|---|---|---|---|
| BERT-base **b1** | 3395 | 19.6 ms | **32.3 %** | 0 | 12.0 |
| BERT-base **b4** | 5820 | 53.2 ms | **64.0 %** | 0 | 14.5 |
| BERT-base **b8** | 5820 | 116.1 ms | **50.7 %** | 0 | 18.5 |
| BERT-base **b16** | 5820 | 140.9 ms | **74.6 %** | 0 | 18.5 |

→ 即便 b16，CUBE 利用率也只有 **74.6%**；`int8_ratio` 全 0 → **从未测过 INT8**；b1 仅铺 12/≈24 核。

### 2.2 AIC pipe 分解（`pipe_baseline_per_model.json`，`source=msprof_PipeUtilization_measured`，µs）

| BERT-base | mac(算) | mte1 | **mte2(HBM)** | fixpipe | 主导 |
|---|---|---|---|---|---|
| b1 | 192.0 | 223.8 | **516.4** | 75.6 | **mte2** |
| b4 | 654.4 | 503.2 | **889.2** | 296.2 | **mte2** |
| b8 | 1107.5 | 1195.7 | **1767.2** | 540.9 | **mte2** |
| b16 | 2054.0 | 1658.8 | **2402.1** | 924.1 | **mte2** |

→ **所有 batch 下 AIC 瓶颈 pipe 恒为 mte2（HBM 访存）**，不是 mac。b1 时 HBM 时间是算力的 2.7×。
**这是不依赖任何模型的实测铁证：BERT/ET-BERT 本质访存受限。**

### 2.3 host/调度开销 + 端到端时延

- **host gap**（msprof，BERT-base b1）：`host_gap_us_per_kernel=41.66`，`n_kernels=338`。
- **ais_bench 干净时延**（200 次，FP16）：BERT-base b1/4/8/16 = **1776.7 / 2368.8 / 3820.6 / 4944.0 µs**
  （×16 batch 时延仅涨 2.8× → 巨大固定开销 ≈ 0.45 ms，即 host 调度墙）。
- **PRISM 校准验证**：roofline 预测 b1 = **1776.2 µs** vs 实测 **1776.7 µs**；regime = **调度受限（β 主导）**，
  T_overhead=1776 ≫ T_compute=114、T_memory=316 µs。

---

## 2.5 ET-BERT 直接实测（910B4，FP16 vs INT8，2026-06-05）

**首次在真机直接实测 ET-BERT 本体**（vocab=256 OM，非 BERT-base 代理）。
环境：Ascend 910B4 服务器（8× 910B4），CANN+ais_bench+msprof；INT8 经 AMCT PTQ（amct_onnx 0.23.2）量化。
**OM 体积**：FP16 **165 MB** / INT8 **85 MB**（恰好 ½）→ 实测坐实「16 MB Cache 装不下整模型」（单层 INT8≈7 MB）。
原始数据：`data/calibration/etbert_measured.json`。

| prec | B | 时延ms | cube_fp16% | **cube_int8%** | mac | mte2 | fix | 主导 | **LLC读%** | **DDR%** |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | 1 | 1.76 | 32.2 | 0 | 189 | **508** | 74 | mte2 | 46.0 | 0.39 |
| FP16 | 16 | 4.93 | 74.5 | 0 | 2055 | **2403** | 926 | mte2 | 47.0 | 0.43 |
| INT8 | 1 | **2.21** | 0 | **14.3** | 84 | **344** | 196 | mte2 | 46.4 | 0.32 |
| INT8 | 16 | **5.35** | 0 | **45.5** | 892 | **1470** | 1152 | mte2 | 44.6 | 0.52 |

（mac/mte2/fix = 每次推理 µs，已÷loop；完整 4×batch 见 JSON）

**七条实测结论：**

1. **代理假设被真机坐实**：ET-BERT FP16 时延 1.76/2.36/3.77/4.93 ms（b1/4/8/16）与 BERT-base 旧实测
   1.78/2.37/3.82/4.94 ms **1% 内吻合**；cube_util 32.2/74.5% 与 BERT-base 32.3/74.6% 几乎逐位相同。
2. **INT8 实测落地（填补 ③）**：`cube_util_int8` 14.3→45.5%（FP16 时代恒为 0）→ 量化确实跑在 INT8 cube。
3. **mac 时间 INT8≈½ FP16**（189→84 / 2055→892）→ 实测确认 INT8 算力 2× 吞吐。
4. **Q3 被实测证实**：同 batch 下 **INT8 的 cube 利用率 < FP16**（b16: 45.5% vs 74.5%；b1: 14.3% vs 32.2%）
   —— 精度↓ 让 CUBE 更早干完、空闲更多、MFU 下降。
5. **mte2(访存) 在 FP16/INT8 所有 batch 恒为主导** → 访存受限本质不变（实测铁证）。
6. **🔥 反直觉杀手锏（只有实测能看见）**：**INT8 比 FP16 更慢**（b1 2.21 vs 1.76 ms，每个 batch 都慢）。
   根因：ET-BERT 非算力受限，砍 mac 不缩 wall-clock；而量化引入 requant 开销（**fixpipe 74→196 µs ↑**、kernel 增多）反而加时延。
   → **实测版 Q3 结论**：在 host/访存受限的模型上，降精度不仅不提速、反而更慢——必须先拆调度/访存墙才能兑现低精度。
7. **填补 ①②**：**LLC(L2) 读命中率 ≈ 46%**（FP16/INT8 接近）；**DDR/HBM 利用率 ≈ 0.4%（近乎空闲）**。
   实测访存层级带宽：**HBM 1.4 → L2 2.7 → L1 122 GB/s**。

> 🔧 **实测修正 PRISM**：PRISM 把 L2 当 scratchpad、假设「权重>L2 即每次从 HBM 流式重载」→ 预测 HBM 受限。
> 但真机显示**稳态（暖 cache，loop=20）下 96 MB L2 + 跨调用复用吸收了几乎全部权重流量，DDR 仅 0.4% 利用**，
> 瓶颈是片上 mte2(L1 填充)+host 调度，**不是 DDR**。（冷启动单次推理仍会从 HBM 冷载 165 MB 一次。）
> 这条只有实测能发现，PRISM 的 `bytes_total=weight-L2` 溢出模型应据此下修稳态 HBM 流量。

---

## 2.6 推演：下一代 fixpipe 吸收量化开销（去 quant/dequant 后重测，2026-06-05）

**假设**：下一代芯片专门为低精度设计 fixpipe 电路，让量化/反量化"免费"。
**做法**（忠实于真机）：不硬改 ONNX（去 quant 后 INT8 matmul 拿不到量化输入、不可运行），
而是按 **OP Type 分解真机 per-op 数据**，把量化税算子剥离，再只在非税算子上重新聚合 LLC/带宽/时延。
新增实测：INT8 的 L2Cache pass（per-op 命中率）。脚本：`decompose2.py` / `reaggregate_nonquant.py`。

**量化税分解（每次推理 µs，profiled）**：

| | 核心 matmul | **税:Quant/Cast/TransData** | **灰区:融合dequant** | 向量 | 税占比 |
|---|---|---|---|---|---|
| FP16 b1 | 754 | 111 | 9 | 1241 | **5%** |
| INT8 b1 | 764 | **494** | **433** | 949 | **19%（+灰 35%）** |
| INT8 b16 | 2244 | 792 | 1294 | 2244 | **12%（+灰 32%）** |

> 量化税 = `AscendQuant`+`Cast`+`TransData`；灰区 = INT8 专有的 `AutomaticBufferFusionOp`（融合反量化）。
> FP16 仅 ~5%（NZ 格式固有 TransData），INT8 高达 19–35%——这就是 INT8 反而更慢的根因。

**去量化税后重测（real per-op 重聚合）**：

| prec | B | DDR利用%(全/去税) | **L2读命中%(全=去税)** | 时延ms 实测→去quant→去quant+dequant |
|---|---|---|---|---|
| FP16 | 1 | 0.39 / 0.39 | 73.9 | 1.758 → 1.666 → 1.658 |
| INT8 | 1 | 0.32 / 0.32 | **99.8** | 2.208 → 1.795 → **1.433** |
| INT8 | 16 | 0.52 / 0.52 | **97.6** | 5.353 → 4.708 → **3.654** |

**三条推演结论：**

1. **量化税是计算/流水开销，不碰 DDR**：去税前后 DDR 利用率不变（0.32→0.32%）→ 下一代 fixpipe 优化的是
   pipeline 延迟，**不改变带宽需求**（带宽结论 Q2 不受影响）。
2. **低精度反而提高 L2 命中率**：INT8 的 L2 读命中 **95–99.8%** > FP16 74–96%（权重减半 85 MB 更易驻 96 MB L2）。
   小 batch 尤甚（b1：INT8 99.8% vs FP16 73.9%）→ 这正是 DDR 近空闲的原因；**精度↓→cache 命中↑→带宽更省**。
   （注：此为 AIC L2 读端口命中率，per-op 实测；§2.5 的系统 `llc_read_write` 计数器 46% 是另一更窄的 LLC 结构，两者口径不同。）
3. **下一代 fixpipe 让 INT8 翻盘**：吸收 quant+dequant 后，**INT8 b1 1.43 ms < FP16 1.76 ms（快 ~18%）**、
   b16 快 ~26%；若只吸收 quant（不含融合 dequant）则与 FP16 基本打平。但即便"免费量化"，加速也只有 ~18–26%
   （**远不到 2× 算力**）——因 ET-BERT 仍受 host/L1(mte2) 约束。**结论：下一代 fixpipe 把 INT8/FP4 从"负收益"
   救回"正收益"，但要吃满低精度算力，仍须同时拆 host 调度墙 + 提 L1/访存带宽。**

> 推演口径：理想时延 = 干净时延 ×(1−税占比)，税占比取自 profiled PipeUtilization 的 OP-Type 分解；
> 单流串行、host gap 主导，逐核减时长为一阶近似。

---

## 3. 四个问题的结论

### Q1：FP8/FP4 混合精度 MFU 怎么公平算？

峰值随精度变，混合精度**没有单一峰值**。PRISM `eta_real/extract.py` 的公平做法 = **按时间加权、各算子用各自精度峰值归一**：

```
MFU_mixed = Σ_op (t_op/T) × util_op      （util_op 已相对该算子精度峰值归一）
cube_util_total = cube_util_fp16 + cube_util_int8   ← 代码原样
```

不要混峰值当分母；跨精度比较应改比 **有效 TOPS / 时延**，或拆成结构因子
`η = η_pipeline·η_tile·η_mem·(1+γ_B·log2 B)`（与峰值无关，看损失在哪层）。
**实测佐证**：b16 实测 CUBE 仅 74.6%、b1 仅 32.3%（§2.1）。

### Q2：800T FP4 与 200T FP8 访存带宽需求一样吗？——**不一样**

| 口径 | FP8(200T)→FP4(800T) | 原因 |
|---|---|---|
| 搬运字节量(MB) | FP4 = **½** × FP8 | 位宽减半 |
| 维持峰值所需带宽(GB/s) | FP4 = **2×** × FP8 | 算力 4× 只被密度 2× 抵消一半 |

```
BW_需求 ∝ 峰值算力 × 位宽
FP8→FP4：4 × ½ = 2×（翻倍）   ｜   910B FP16→INT8：2 × ½ = 1×（不变！）
```

PRISM 实算（ET-BERT，维持 compute-bound 的最小带宽）：**FP8 74 GB/s → FP4 147 GB/s（正好 2×）**；
910B FP16/INT8 两档都是 232 GB/s（不变）。**带宽需求是否变化 = 算力倍数 vs 密度倍数是否相等。**

### Q3：FP8→FP4，MFU 会下降、需更大带宽吗？——**会，主导是 roofline 左移而非 tile 变大**

PRISM 校准模型（含 host 开销）**实算 cube-busy%（≈MFU）随精度逐档减半**：

| batch | FP16 | INT8/FP8 | FP4 | 受限(FP16) |
|---|---|---|---|---|
| 1 | 6.4% | 3.2% | 1.6% | host 调度 |
| 16 | 33.9% | 17.0% | 8.5% | 调度/带宽 |
| 256 | 45.7% | 23.2% | 11.6% | 内存 |

机理：① **roofline 左移（主导）**——FP4 算力 4×，T_compute 减半但访存/host 不变 → CUBE 被饿、空闲翻倍；
② **tile 变大（对 ET-BERT≈0）**——原生 K-block 随精度加深（16→32→64），但 ET-BERT 归约维全是 64 倍数，
`η_tile=1.000`（K0=16/32/64 三档都不变）。**实测佐证**：AIC 主导恒为 mte2（§2.2）→ 加带宽才能减空闲，与 Q2 一致。

### Q4：16MB Cache 能兜 30MB 模型？加 Cache 降带宽？DDR 利用率？精度变化？对齐？

- **兜不住整模型，但不需要**：推理逐层流式，关键是**单层工作集**（ET-BERT FP16 **14.16 MB**），
  **16MB 恰好兜住一层 FP16**。低于 ~14MB（如 8MB）→ tile 回取放大（`archetype_amplification` 1.06–14×）。
- **加 Cache 只在"兜住单层"前有效**（PRISM L2 扫描）：8→16→32→96MB 后 DDR 流量饱和；单流推理 Cache 超单层无进一步收益（不会回用上层权重）。更大 Cache 的价值在 batch/多请求跨请求复用。
- **DDR 利用率 / LLC 命中率**：⚠️ 仓库**空白**，须跑 `run_etbert_llc_ddr_msprof.sh` 实测（PRISM 预测 b1 低 ~18%、b256 才饱和）。
- **精度变化后带宽**：绝对流量 FP8→FP4 减半；但 DDR **利用率上升**（算力 4×、流量仅 ½ → 更易 memory-bound）。
- **cacheline 对齐**：910B fractal 按 **512B 对齐**，低精度加深内维保持 512B（FP16 16×16 / INT8 16×32 / FP4 16×64）。
  只要归约维是 32/64 倍数即零浪费——**ET-BERT 各维全 64 倍数，FP4 下仍完全对齐**。风险在非对齐维（奇异 d_head/seq）。
  PRISM 按精确字节计数、**不建模 fractal padding**，须由实测补对齐损失。

---

## 4. 新增工具（本备忘录配套，可直接命令行复现）

### 4.1 ET-BERT 模型 + 精度档（`prism-predict-pipe`）

新增 `models/regime/et_bert.yaml`，并给 `prism-predict-pipe` 加 `--precision {fp16,int8,fp8,fp4}`
（只缩放 CUBE/GEMM 路径：权重+激活按位宽、mac 按吞吐；AIV/host 保持 FP16——符合真实混合精度量化）：

```bash
for p in fp16 int8 fp8 fp4; do
  prism-predict-pipe --model models/regime/et_bert.yaml \
    --arch arch/ascend_910b4_for_sweep_v2.yaml --batch 1 --precision $p \
    --output data/calibration/predict_pipe_et_bert_$p.json
done
# 非对称下一代(FP4 800T / FP8 200T，算力 4× 密度 2×)：
prism-predict-pipe --model models/regime/et_bert.yaml --precision fp4 --mac-mult 4 ...
```

**PRISM 实算（ET-BERT b=1，aic_pipes µs）**——验证精度缩放与"mte2 恒主导"：

| precision | byte_factor | mac_mult | mac | **mte2(HBM)** | dominant | wall |
|---|---|---|---|---|---|---|
| fp16 | 1.0 | 1.0 | 233.9 | **567.3** | mte2 | 15492 |
| int8 / fp8 | 0.5 | 2.0 | 117.0 | **249.7** | mte2 | 15174 |
| fp4 | 0.25 | 4.0 | 58.5 | **124.9** | mte2 | 15050 |

→ mac/mte2 逐档减半、dominant 恒为 mte2、wall 几乎不变（host 受限）——与实测方向一致。

### 4.2 两段 msprof 采集脚本（补齐仓库三项空白）

| 脚本 | 补齐 | 关键指标 |
|---|---|---|
| `benchmark/run_etbert_llc_ddr_msprof.sh` | ① LLC/L2 命中率 ② DDR 带宽利用率 | `--aic-metrics=L2Cache/Memory` + `--sys-hardware-mem=on` |
| `benchmark/run_etbert_int8_msprof.sh` | ③ INT8 实测（vs FP16） | AMCT PTQ→ATC→`ArithmeticUtilization/PipeUtilization/L2Cache` |

两脚本均遵循仓库既有约定（`ais_bench --loop/--warmup_count`、`--l2=on`、op_summary CSV、rsync 取回），
脚本尾注明**解析口径与预期对照**，结果回填后即把本备忘录 Q4/实测桥由"PRISM 预测"升级为"910B 实测"。

---

## 5. 一句话总览

ET-BERT 在 910B **小 batch 是 host 调度墙**（实测 CUBE 仅 32%@b1）、**AIC 恒由 mte2 主导**（实测，FP16/INT8 皆然）；
**实测 INT8 比 FP16 更慢、cube 利用率更低**（降精度在访存/host 受限模型上不提速反而更差）——这正是 FP8→FP4 的真机预演；
FP8→FP4 算力 4× 但密度仅 2× → **带宽需求翻倍、MFU 逐档减半**；16MB Cache 装不下整模型（实测 OM 165/85 MB），
但稳态下 **96 MB L2 已让 DDR 近乎空闲（实测 0.4%、LLC 命中 46%）**；fractal 512B 对齐对 ET-BERT 整齐维度在 FP4 下仍无损。
**要兑现 FP4：先拆调度/访存墙（融合+大 batch），再上 2× 带宽。**
三项空白（INT8/LLC/DDR）已于 2026-06-05 在 910B4 真机**实测补齐**（`data/calibration/etbert_measured.json`），FP8/FP4 由 `prism-predict-pipe --precision` 外推。
