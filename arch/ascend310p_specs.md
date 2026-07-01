# 昇腾 310P 规格参数（校准基准）

> 来源：官方 Atlas 300I Duo 文档 + Kimi DeepResearch 架构分析 + 用户实测确认
> 更新日期：2026-04-27

## 核心规格

| 参数 | 数值 | 可信度 | 来源 |
|---|---|---|---|
| INT8 算力 | **~140 TOPS**（整卡）| 中（双芯片280÷2推算）| Atlas 300I Duo 官方文档 |
| FP16 算力 | **~70 TFLOPS**（整卡）| 中 | Atlas 300I Duo 官方文档 |
| 单 Core INT8 | **16 TOPS**（@ ~1.1 GHz）| 高 | 架构分析推算 |
| 单 Core FP16 | **8 TFLOPS** | 高 | 架构分析推算 |
| AI Core 数 | **8 个** | 高 | 官方文档 |
| 内存带宽 | **204.8 GB/s**（384-bit LPDDR4X）| 高 | 用户确认 |
| 内存容量 | **48 GB / 芯片** | 高 | 拆解确认 |
| 功耗 TDP | **~150 W / 芯片** | 高 | Atlas 300I Duo 整卡300W÷2 |
| 制程工艺 | 未公开（推测12-16nm）| 低 | 业界推测 |

## 完整片上存储层次（用户确认 + Kimi DeepResearch）

| 存储层级 | 容量 | 管理类型 | 关联单元 | 估算带宽 | 核心功能 |
|---------|------|---------|---------|---------|---------|
| L0A / L0B | 1–4 KB（各）| Scratchpad | Cube 输入 | ~1–2 TB/s | 矩阵乘法双输入缓冲 |
| L0C | 2–8 KB | Scratchpad | Cube 输出 | ~1–2 TB/s | 矩阵乘法输出/累加缓冲 |
| UB | **256 KB / core**（2MB总）| Scratchpad | Vector 专用 | ~500GB–1TB/s | 向量运算工作集 |
| L1 Buffer | **1 MB / core**（8MB总）| Scratchpad | Cube 输入暂存 | ~200–500 GB/s | 权重/激活分块中转、格式转换 |
| L2 Buffer | **16 MB**（芯片级共享）| Scratchpad | 全 Core 共享 | ~100–200 GB/s | 跨层权重缓存、跨 Core 共享 |
| DDR（LPDDR4X）| 48 GB | — | 片外主存 | **204.8 GB/s** | 模型权重、激活存储 |

**片上存储合计：~26 MB**（L0 ~12KB + UB 2MB + L1 8MB + L2 16MB）

## DaVinci 微架构数据流（用户确认）

```
Global Memory (LPDDR4X, 48GB, 204.8 GB/s)
    │
    ▼
L2 Buffer (16MB, 芯片级共享 Scratchpad)
    │
    ├──► L1 Buffer (1MB/core × 8, Cube 输入暂存 + 格式转换)
    │        │
    │        ├──► L0A (输入矩阵 A, 1-4KB) ┐
    │        └──► L0B (输入矩阵 B, 1-4KB) ├──► CUBE (16×16×16 脉动阵列) ──► L0C (2-8KB)
    │                                      ┘                                      │
    │                                                                              │ (L0C→UB)
    └──► UB (256KB/core × 8, Vector 专用) ◄───────────────────────────────────────┘
             │                                    GEMM结果→Vector做LayerNorm/Softmax
             ▼
         Vector 引擎 (128 FP16 OP/周期)
             │
             ▼
            UB ──► L2 ──► Global Memory
             │
             └──► L1 (UB↔L1互通: Vector处理后结果再次输入Cube做下一轮计算)
```

**Cube 路径**：Global Memory → L2 → L1 → L0A/B → Cube → L0C → UB → L2 → Global Memory

**Vector 路径**：Global Memory → L2 → UB → Vector → UB → L2 → Global Memory

**跨路径交互**：
- L0C ↔ UB：GEMM 中间结果送 Vector 做 LayerNorm/Softmax
- UB ↔ L1：Vector 处理后结果再次送 Cube 做下一轮矩阵乘

## 计算单元规格

| 单元 | 结构 | 每周期操作数 | 峰值算力（单Core）| 精度 |
|---|---|---|---|---|
| **Cube** | 16×16×16 3D 脉动阵列 | 4096 MAC（FP16）/ 8192 MAC（INT8）| 8 TFLOPS / 16 TOPS | FP16, INT8 |
| **Vector** | SIMD | **128 OP（FP16）** | ~256 GFLOPS | FP16, FP32, INT8 |
| **Scalar** | 顺序执行 | ~1-2 指令 | ~2 GIPS | INT32, INT64 |

**关键比例**：Cube : Vector = 4096 : 128 = **32:1**
→ Vector 是 Transformer 推理中 LayerNorm/Softmax 的潜在瓶颈

## Roofline 关键参数

```
Ridge Point（Cube）= 140 TOPS / 204.8 GB/s ≈ 683 OPs/Byte

AI-DPI 负载（~50MB Transformer，batch=1-8）：
  OI（无L2命中）= 200M FLOPs / 50MB ≈ 4 OPs/Byte    → 严重带宽瓶颈
  OI（L2命中32%）= 200M FLOPs / 34MB ≈ 6 OPs/Byte   → 仍严重带宽瓶颈
  OI（权重常驻）= 200M FLOPs / 196KB ≈ 1000 OPs/Byte → 转为算力瓶颈

有效权重缓存 = L2 16MB（L1 8MB 专用于 Cube tile 暂存，不适合跨层缓存）
50MB 模型 L2 命中率 ≈ 32%，Cube 利用率 < 1%
```

## Vector 瓶颈分析（新增）

```
Transformer 中 Vector 操作占比估算（batch=1）：
  LayerNorm：2× per layer × 6 layers = 12 次
  Softmax：1× per layer × 6 layers = 6 次
  GELU/ReLU：1× per layer × 6 layers = 6 次

Vector 吞吐 = 128 OP/cycle × ~1.1 GHz ≈ 141 GOPS
Cube 吞吐  = 16 TOPS（INT8）

→ 若 Vector 操作占总 FLOPs 的 3% 以上，Vector 成为瓶颈
→ Transformer 中 LayerNorm+Softmax 约占 5-10% FLOPs
→ 定制芯片需提升 Vector/Cube 比例（当前 1:32 过于悬殊）
```

## Timeloop 建模关键约束

| 约束 | 数值 | 说明 |
|---|---|---|
| Cube 最小分块 | **16×16×16** | 硬件粒度，tiling 必须为 16 的整数倍 |
| Cube 绑定存储 | L0A/B/C | Cube 指令只能访问 L0，不能直接访问 UB |
| Vector 绑定存储 | UB（256KB）| Vector 指令只能访问 UB |
| 跨单元数据传递 | 必须经 L1/L2 中转 | L0C→UB 或 UB→L1 需 MTE 搬运 |
| MTE | 独立异步 DMA | 与计算单元并行，建模为独立资源 |
| UB 对齐要求 | 32 字节对齐 | 未对齐访问有额外开销 |
| 双缓冲 | Ping-Pong | 有效容量减半，但隐藏搬运延迟 |

## 定制目标规格（选项B）

| 参数 | 310P 现状 | 定制目标 | 调整逻辑 |
|---|---|---|---|
| INT8 算力（Cube）| 140 TOPS | **16-32 TOPS** | batch=1-8 下利用率<1%，大幅削减 |
| 内存带宽 | 204.8 GB/s | **256-512 GB/s** | 首要瓶颈 |
| L2（权重缓存）| 16 MB | **32-64 MB** | 使 50MB 模型完整常驻，OI 从 6 跳至 1000 |
| Vector 吞吐 | 128 OP/cycle | **提升 4-8×** | LayerNorm/Softmax 当前是潜在瓶颈 |
| 功耗 | ~150 W | **≤30-50 W** | 线卡集成硬性约束 |
| Vector/Cube 比 | 1:32 | **1:8 ~ 1:16** | Transformer 推理中 Vector 占比更高 |

## 待补充

- ByteMLPerf 实测归一化延迟曲线（batch=1/4/8/16）
- α/β 校准系数
