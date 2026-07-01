# 教程 1：30 分钟 quickstart

从 git clone 到看到第一份 finding 报告，约 30 分钟。

## 0. 前置

- Python 3.9+
- Git
- 推荐用 macOS 或 Linux（NPU 实机仿真 NPU-side 工作不在本教程范围）

## 1. Clone + 创建虚拟环境（5 min）

```bash
git clone <repo>
cd <repo>

python3 -m venv .venv
source .venv/bin/activate     # bash/zsh
# 或：source .venv/bin/activate.fish for fish
```

## 2. 安装包（3 min）

```bash
pip install --upgrade pip       # 确保 pip ≥ 23（PEP 660 editable install）
pip install -e ".[dev]"
```

预期看到 7 个 CLI 入口安装：
```
prism-extract  prism-fit  prism-regime  prism-sweep  prism-ceiling  prism-mapping  prism-render
```

验证：
```bash
which prism-sweep         # → .venv/bin/prism-sweep
prism-sweep --help        # 看到帮助即 OK
```

## 3. 跑 baseline 工具（5 min）

```bash
prism-render --check     # 4 份 finding 报告与已 commit 一致 → exit 0
prism-sweep              # 12 维 49 变体 × 5 models 架构 sweep
prism-ceiling            # 5 情景 × 11 配置优化天花板
```

期望输出：

| CLI | 输出 |
|-----|------|
| `prism-render --check` | `4 OK identical` |
| `prism-sweep` | terminal 显示 49 变体 × 5 models 的 ratio 表 + 写入 `data/outputs/phase_j_sweep.json` |
| `prism-ceiling` | 5 情景 × 11 配置 + 写入 `data/outputs/optimization_ceiling.json` + `docs/findings/optimization_ceiling.md` |

## 4. 看 4 份 finding 报告（10 min）

打开浏览器或 markdown 查看器，按顺序读：

```bash
open docs/findings/主报告.md                  # 最高优先级，立项核心论证
open docs/findings/optimization_ceiling.md   # 算子/软件/硬件 5 情景天花板
open docs/findings/微架构探索报告.md          # 12 维架构 sweep 结果
open docs/findings/roofline校准报告.md        # wall-clock 三层模型 + Timeloop 验证
open docs/findings/msprof分解报告.md          # msprof 实测 wall-clock vs NPU compute gap
```

## 5. 跑 tests（2 min）

```bash
pytest tests/ -v
```

期望：全绿（5 unit + 1 E2E test）。

## 6. 看方法论文档（5 min）

打开总览：

```bash
open docs/methodology/01_overview.md
```

按建议顺序读：
- 01 总览 → 02 三层 Roofline → 04 sweep 方法论 → 07 ceiling

约 5 分钟翻完核心方法论。

---

## 完成后你应该理解

- 工具的 7 个 CLI 各自做什么
- 4 份 finding 报告的含义
- baseline 重现的不变量（render --check / sweep / ceiling 跑出与 reference 一致）
- 加新模型 / 加新 sweep 维度 的入口

## 下一步

| 你的目的 | 教程 |
|---------|------|
| 复现完整 49 维 sweep | [02_reproduce_arch_sweep.md](02_reproduce_arch_sweep.md) |
| 用新 msprof 数据重新校准 η_real | [03_recalibrate_with_new_msprof.md](03_recalibrate_with_new_msprof.md) |
| 加新 transformer 模型 | [04_add_new_model.md](04_add_new_model.md) |
| 加新架构 sweep 维度 | [methodology/04_arch_sensitivity.md §9](../methodology/04_arch_sensitivity.md#9-加新-sweep-维度) |
| 加新优化情景到 ceiling | [methodology/07_optimization_ceiling.md §8](../methodology/07_optimization_ceiling.md#8-添加新优化情景) |

---

## 故障排除

### `pip install -e .` 失败：editable mode requires setuptools-based build

→ pip 太老（< 23）。两条路：
- 升级 pip：`python3 -m pip install --upgrade pip`
- 用本工具的 `setup.py` shim：`pip install -e .` 应自动使用 setup.py 兜底

### `prism-render --check` 报 DIFF

→ 模板与 vars JSON 不同步。可能是：
- 你修改了 template 但忘了 commit 渲染输出
- vars JSON 改了但模板没引用对应 var

修复：
```bash
prism-render          # 重新渲染（不用 --check）
git diff docs/findings/    # 看 diff
git commit ...      # 提交渲染结果
```

### 模块 import 报错（如 `ModuleNotFoundError: prism`）

→ 没装包或没 activate venv：
```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

或用 no-install 路径：
```bash
python3 scripts/prism_render.py --check
```

### `pytest` 报"no tests collected"

→ M4 阶段添加测试中。当前 tests/ 是骨架，CI E2E 校验通过 prism-render --check + prism-sweep 重现一致性间接验证。
