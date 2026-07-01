# 教程 2：复现 12 维架构 sweep

复现 [docs/findings/微架构探索报告.md](../findings/微架构探索报告.md) 中 49 变体 × 5 模型的 ratio 表。

---

## 0. 前置

- 已完成 [tutorials/01_quickstart.md](01_quickstart.md)
- `data/calibration/pipe_baseline_per_model.json` 存在（应当 git tracked）

## 1. 准备 baseline pipe data

如 `data/calibration/pipe_baseline_per_model.json` 已 commit，跳到 §3。否则需 bootstrap：

```bash
# 从 NPU 服务器 rsync 拉回 msprof 原始数据
rsync -avz user@npu-server:~/sim-experiment/msprof_data/ msprof_data/

# 提取 pipe baseline JSON
prism-extract --output data/calibration/pipe_baseline_per_model.json
```

期望 JSON 含 11 个 configs：

```python
import json
data = json.load(open('data/calibration/pipe_baseline_per_model.json'))
print(list(data['configs'].keys()))
# ['BERT-base-S128-b1', 'GPT-2-S512-b1', 'Net-Transformer-S256-L1-b1',
#  'Qwen3-prefill-S256-b1', 'Qwen3-prefill-S256-b4', 'Qwen3-prefill-S256-b8',
#  'Qwen3-prefill-S512-b4', 'Qwen3-prefill-S512-b8',
#  'Qwen3-prefill-S4096-b1', 'Qwen3-Embedding-S4096-b1',
#  'Qwen3-decode-Min4-Skv128-b1']
```

## 2. 准备 arch baseline yaml

`arch/ascend_910b4_for_sweep_v2.yaml` 是 sweep 默认 baseline，已 git tracked。验证：

```bash
ls arch/ascend_910b4_for_sweep_v2.yaml
```

如不存在，用 `arch/ascend_910b4.yaml` 派生（手动复制 + 加 12 个 sweep 字段）。详见 [reference/arch_yaml_schema.md](../reference/arch_yaml_schema.md)。

## 3. 跑 sweep

```bash
prism-sweep \
  --pipe-baseline data/calibration/pipe_baseline_per_model.json \
  --arch-baseline arch/ascend_910b4_for_sweep_v2.yaml \
  --output        data/outputs/phase_j_sweep.json
```

约 5-10 秒（pipe-aware analytical，无 Docker）。

终端输出（截取）：

```
                       BERT  GPT-2  Qwen3-pre  Qwen3-Emb  Qwen3-dec  NetT
n_cores=8            | 1.00  1.00     2.95       2.95       1.85      1.00
n_cores=16           | 1.00  1.00     1.50       1.50       1.20      1.00
n_cores=24 (baseline)| 1.00  1.00     1.00       1.00       1.00      1.00
n_cores=48           | 1.00  1.00     0.84       0.84       0.92      1.00
ub_l1_fused=True     | 0.99  1.00     0.80       0.80       0.93      1.00
hbm_bw=800 (HBM3)    | 1.00  1.00     0.85       0.85       0.65      1.00
...
```

## 4. 与 reference 对比

工具的 reference output 是 git-tracked 的 [docs/findings/微架构探索报告.md](../findings/微架构探索报告.md)。验证一致性：

```bash
prism-render --check   # exit 0 表示 4 模板渲染与 commit 一致
```

如 sweep 改了某 variant，会让 vars JSON 变 → 报告渲染变 → render --check 报 DIFF。

## 5. 看 sweep 完整结果

```bash
# JSON 数据
cat data/outputs/phase_j_sweep.json | python3 -m json.tool | head -100

# 人读报告
open docs/findings/微架构探索报告.md
```

## 6. 自定义 sweep（加新 variant）

例：测试 "n_cores=64" 是否带来收益。

编辑 `src/prism/sweep/runner.py` 顶部 `SWEEP` dict：

```python
SWEEP = {
    'n_cores': [8, 12, 16, 24, 32, 48, 64],   # 加 64
    ...
}
```

重跑：

```bash
prism-sweep
# 终端会多一行：
# n_cores=64           | 1.00  1.00     0.84       0.84       0.92      1.00
# 跟 n_cores=48 一样饱和（host_gap floor 已封顶）
```

## 7. 自定义 baseline arch（如换 LPDDR5X 试一组对照）

复制 baseline yaml 修改：

```bash
cp arch/ascend_910b4_for_sweep_v2.yaml arch/ascend_910b4_lpddr5x.yaml
# 编辑 arch/ascend_910b4_lpddr5x.yaml: hbm_bw_gbs: 100, l2_mb: 32

prism-sweep --arch-baseline arch/ascend_910b4_lpddr5x.yaml \
          --output        data/outputs/sweep_lpddr5x_baseline.json
```

→ 现在 sweep 中所有 ratio 是相对 "LPDDR5X baseline" 的，而不是 910B4。**注意基线变化会影响 ratio 解读**。

## 8. 常见问题

### 整 column ratio = 1.00（如 Net-Transformer）

→ 该 model 是 host_gap 主导（host_pct > 90%）。改 device-side 资源的 sweep 维度都不影响 wall_clock。这是物理事实，不是 bug。

### `pipe_baseline_per_model.json` 中缺 key

→ sweep 跑时会跳过缺失的 model。补充 model 用 `prism-extract`，详见 [tutorials/04_add_new_model.md](04_add_new_model.md)。

### sweep 给的 variant 在你想要的范围之外

→ 编辑 `src/prism/sweep/runner.py` 的 `SWEEP` dict（每维度数组）。

### 想测联合 sweep（如 n_cores × HBM_bw 全组合）

→ 当前工具是单维度独立 sweep。如要联合 sweep 需改 `runner.py` 添加 cartesian product 循环，详见 [methodology/04_arch_sensitivity.md §2.2](../methodology/04_arch_sensitivity.md#22-每维度独立改变不做联合-sweep)（关于联合 sweep 的设计权衡）。

---

## 完成后你应该能

- 复现 49 变体 × 5 模型 sweep
- 看懂 sweep 输出的 ratio 含义
- 加新 sweep variant 维度
- 改 baseline arch 跑对照

## 下一步

| 目的 | 教程 |
|-----|------|
| 加新模型 | [04_add_new_model.md](04_add_new_model.md) |
| 加新 sweep 维度（如 KV prefetcher）| [methodology/04_arch_sensitivity.md §9](../methodology/04_arch_sensitivity.md#9-加新-sweep-维度) |
| 改 wall-clock 公式假设 | [methodology/02_three_layer_roofline.md §7](../methodology/02_three_layer_roofline.md#7-wall-clock-预测公式-v3) + 06 |
