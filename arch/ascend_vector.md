# 昇腾910B Vector单元直连Unified Buffer的性能分析

## 一、架构背景：Da Vinci的存储层次

昇腾910B采用Da Vinci架构，其片上存储层次如下：

```
┌─────────────────────────────────────────────────────┐
│                   AI Core                           │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Cube Unit│  │Vector Unit│  │Scalar Unit│         │
│  │(矩阵计算) │  │(向量计算) │  │(标量控制) │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │              │              │                │
│  ┌────▼─────┐  ┌─────▼──────────────▼──┐            │
│  │ L1 Buffer│  │   Unified Buffer (UB)  │            │
│  │  (~1MB)  │  │      (~256KB SRAM)     │            │
│  └──────────┘  └───────────────────────┘            │
│                         │                           │
│                  ┌──────▼──────┐                    │
│                  │  L2 Buffer  │                    │
│                  │  (~32MB)    │                    │
└──────────────────┴─────────────┴────────────────────┘
                            │
                          DRAM (HBM)
```

**关键点**：Vector Unit 没有独立的 Register File，直接以 UB（SRAM）作为操作数来源和结果写回目标。

---

## 二、延迟量化对比

### 2.1 典型存储层级延迟（7nm工艺节点估算）

| 存储层级 | 容量 | 访问延迟（cycles） | 带宽 |
|---------|------|-----------------|------|
| **Register File** | ~32KB | **1–2** | 极高（多读写口） |
| **Small SRAM（~8KB）** | ~8KB | 2–4 | 高 |
| **UB SRAM（~256KB）** | 256KB | **6–12** | 宽位宽补偿 |
| **L2 Buffer** | ~32MB | 20–50 | 中 |
| **HBM DRAM** | ~32GB | 200–400+ | 受带宽限 |

### 2.2 Vector操作的实际延迟影响

以一条 `VADD`（向量加法）为例：

```
【传统 Register-based（如GPU CUDA Core）】
  读操作数 A: 1 cycle（Register）
  读操作数 B: 1 cycle（Register）
  执行计算:   1–4 cycles（pipeline）
  写结果:     1 cycle（Register）
  ──────────────────────────────
  总延迟:     ~3–6 cycles

【昇腾 UB-based（Da Vinci Vector Unit）】
  读操作数 A: 6–12 cycles（UB SRAM）
  读操作数 B: 6–12 cycles（UB SRAM，可并行）
  执行计算:   1–4 cycles（pipeline）
  写结果:     6–12 cycles（UB SRAM）
  ──────────────────────────────
  总延迟:     ~13–28 cycles（最坏情况串行）
```

### 2.3 吞吐量 vs 延迟的分离

```
                 延迟(Latency)        吞吐(Throughput)
Register方案      低（~5 cycles）      高
UB方案            高（~20 cycles）     受pipeline深度影响，
                                      软件流水化后可接近理论峰值

关键结论：
  单条指令延迟差距 ≈ 3–5x 更差
  但流水线满载后吞吐差距 ≈ 10–30%（可通过软件弥补）
```

---

## 三、为何昇腾选择UB而非Register File？

这是一个**设计权衡**，有其深层原因：

| 维度 | Register File | Unified Buffer |
|------|--------------|---------------|
| 容量 | 小（KB级） | 大（256KB+） |
| 灵活性 | 静态分配 | 动态切片 |
| 编译复杂度 | 高（寄存器分配） | 低（地址管理） |
| 多算子共享 | 难 | 易（统一地址空间） |
| 面积/功耗 | 多端口代价高 | 可优化 |

UB本质上是一个**软件管理的暂存器**，把寄存器分配的复杂性转移给编译器（CANN/TBE），换取了更大的片上工作集。

---

## 四、改进措施

### 4.1 硬件层面

**① 宽数据通路（Wide Data Path）**

UB采用超宽访问位宽（512bit ~ 1024bit），单次读取即可填满Vector单元的操作数，以**带宽换延迟**：

```
256KB UB, 512bit 数据通路：
  理论读带宽 = 512bit × f_clock ≈ 512b × 1.8GHz ≈ 115 GB/s（片上）
  远超 HBM 带宽（~900GB/s总，但多核共享）
```

**② 流水线深化（Deep Pipeline）**

Vector Unit内部采用深流水线，将UB访问与计算重叠：

```
时序示意（流水线满载）：
Cycle:  1    2    3    4    5    6    7    8 ...
inst0: [UB_R][UB_R][EXE][EXE][WB ]
inst1:       [UB_R][UB_R][EXE][EXE][WB ]
inst2:             [UB_R][UB_R][EXE][EXE][WB]
                                         ↑
                                    流水满载后吞吐=1 inst/cycle
```

**③ Bank并行（多Bank SRAM）**

UB被设计为多Bank结构，允许同时读取多个不冲突地址：

```
UB
├── Bank 0: 操作数A的数据
├── Bank 1: 操作数B的数据  ← 同周期并行读
├── Bank 2: 暂存区
└── Bank 3: 结果写回区    ← 与读不冲突
```

---

### 4.2 软件/编译器层面（CANN/TBE）

**① 双缓冲（Double Buffering / Ping-Pong）**

最核心的优化手段，将DMA搬运与计算完全重叠：

```python
# TBE 伪代码示意
# UB 分成两块: buf_A, buf_B

# 预加载第0块数据
dma_copy(src=L2[0:tile], dst=UB.buf_A)

for i in range(1, N):
    # 异步加载下一块（后台进行）
    dma_copy_async(src=L2[i*tile:(i+1)*tile], dst=UB.buf_B)
    
    # 同时计算当前块（前台）
    vector_compute(UB.buf_A)
    
    sync()          # 等待DMA完成
    swap(buf_A, buf_B)  # 切换ping-pong
```

**② 指令流水调度（Instruction Scheduling）**

编译器将UB读取指令提前发出，利用乱序窗口隐藏延迟：

```
未优化（串行）:
  LOAD A ──→ LOAD B ──→ VADD ──→ STORE C   （延迟叠加）

优化后（乱序/重排）:
  LOAD A
  LOAD B     ← 提前，与 LOAD A 并行
  ...（其他无关指令填充延迟槽）
  VADD       ← 此时数据已就绪
  STORE C
```

**③ Tiling策略（数据局部性最大化）**

通过精心的循环分块，让热数据留在UB中被反复使用，减少UB的读写次数：

```
对 GEMM: C[M,N] += A[M,K] × B[K,N]

Tiling to UB:
  UB 容量 = 256KB
  tile_M × tile_K × sizeof(fp16) ≤ UB/3  （A、B、C各占1/3）

  → 内层循环完全在UB内完成，不回写L2/DRAM
  → 减少 UB 访问转化为对延迟的倍增
```

**④ 流水线同步原语（Pipeline Barrier）**

CANN提供细粒度同步指令，避免不必要的全局屏障：

```
pipe_barrier(PIPE_V)    # 仅同步 Vector pipe
pipe_barrier(PIPE_MTE)  # 仅同步 DMA pipe
# 而非全局 sync_all()，减少空泡（bubble）
```

---

## 五、综合量化结论

```
性能损失（相比理想Register方案）：

  单条指令延迟:    差 3–5x（约15–25 cycles vs 5 cycles）
  
  有效吞吐（软件优化后）:
    ├── 无双缓冲:    损失 ~30–50%
    └── 双缓冲+指令调度: 损失 ~5–15%，接近理论峰值
  
  实测利用率（昇腾910B Vector）:
    典型算子（Sigmoid/ReLU）: ~60–80% 利用率
    经精调算子（LayerNorm）:  ~85–92% 利用率
```

**一句话总结**：UB直连带来的**延迟代价是真实的（3–5x）**，但通过双缓冲、宽数据通路和编译器指令调度，**吞吐损失可压缩到10%以内**——这也是Da Vinci架构的核心编程挑战所在，大量优化工作沉淀在CANN的TBE算子库中。