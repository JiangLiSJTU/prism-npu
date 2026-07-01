#!/usr/bin/env python3
"""
Phase I/L — Manual Mapping Generator v0.3（含 Phase L#2 修复）

为 Ascend DaVinci 风格架构（910B4 / 310P）生成 Timeloop manual mapping yaml。

v0.3（Phase L#2，2026-05-08）：
  1. L2_timeloop_view_bytes 从 2 MB（CACTI 限制妥协）改为真实 96 MB
     → 配套 arch yaml: arch/ascend_910b4_for_mapping.yaml（depth=50331648）
     → 大 weight tile / 大 K_inner mapping 不再被 1.5 MB Timeloop view 强切

v0.2 vs v0.1 关键改动（基于 docs/ascend_910b_architecture_reference.md）：
  1. L1 input tile 容量校验（Timeloop 看 L1=1 MB，不是 512 KB；CACTI 限制）
  2. K_outer 自动放 L2_Buffer temporal（不放 L0B，因 L0B 持单 K_inner tile 即可）
  3. M_l1 自动切分：M_per_core × K_inner × n_cube × 2 > 1 MB 时拆 M_l1
  4. Spatial split=3（所有因子进 X 方向，匹配 1D arch fanout）
  5. Architecture 参数从 dict 派生，方便扩到 310P / 自研芯片

数据流（910B 含 FixPipe）：
  HBM → L2 → L1 → L0A/L0B → Cube → L0C → (FixPipe → L1 / HBM, 或 → UB → AIV)

Timeloop yaml 中各级 datatype 配置（auto-mapper 风格 + 架构正确）：
  L2_Buffer: keep [Weights]
  L1_Buffer: keep [Inputs]
  L0A:       keep [Inputs]
  L0B:       keep [Weights]
  L0C:       keep [Outputs]
  Cube_Reg:  keep []  (everything bypass — 进 Cube_ALU)
  DRAM:      keep [Inputs, Weights, Outputs]
"""

import argparse
import sys
from pathlib import Path

# ── 默认 Ascend 910B4 架构（v0.3：含 L#1 + L#2 修复）─────────────────────
ARCH_910B4 = {
    "name":             "ascend_910b4",
    "n_cores":          24,                # AICore 数（V2 模式）
    "n_macs_per_core":  4096,              # Cube_MAC[0..4095] 全占用（L#1 K=16 spatial 后）
    "cube_m":           16,                # Cube spatial M (16×16×16)
    "cube_n":           16,                # Cube spatial N
    "cube_k":           16,                # Cube spatial K (L#1 fix: K 也是 spatial via adder tree)
    "l0a_bytes":        65536,             # 64 KB / AIC
    "l0b_bytes":        65536,             # 64 KB / AIC
    "l0c_bytes":        131072,            # 128 KB / AIC
    "l1_timeloop_view_bytes": 1048576,     # Timeloop 看 1 MB（实际 512 KB；depth × 4 cluster）
    "l2_timeloop_view_bytes": 100663296,   # L#2 fix: 96 MB（用 ascend_910b4_for_mapping.yaml）
    "fp16_bytes":       2,
    "fp32_bytes":       4,
}


# ── 容量校验 ────────────────────────────────────────────────────────────
def check_l0_capacity(m_inner: int, n_inner: int, k_inner: int, arch: dict) -> dict:
    """
    L0 per-core 容量校验（Timeloop 的 spatial broadcast worst-case）。
    """
    fp16 = arch["fp16_bytes"]
    fp32 = arch["fp32_bytes"]

    l0a = m_inner * k_inner * n_inner * fp16  # Inputs broadcast 到 N spatial
    l0b = k_inner * n_inner * m_inner * fp16  # Weights broadcast 到 M spatial
    l0c = m_inner * n_inner * fp32             # Outputs

    return {
        "l0a_bytes": l0a, "l0a_pct": 100 * l0a / arch["l0a_bytes"],
        "l0b_bytes": l0b, "l0b_pct": 100 * l0b / arch["l0b_bytes"],
        "l0c_bytes": l0c, "l0c_pct": 100 * l0c / arch["l0c_bytes"],
        "l0a_ok": l0a <= arch["l0a_bytes"],
        "l0b_ok": l0b <= arch["l0b_bytes"],
        "l0c_ok": l0c <= arch["l0c_bytes"],
    }


def check_l1_capacity(m_l1: int, k_inner: int, n_cube: int, arch: dict) -> dict:
    """
    L1 per-core 容量校验（仅 keep Inputs，N 方向 broadcast）。
    """
    l1 = m_l1 * k_inner * n_cube * arch["fp16_bytes"]
    return {
        "l1_bytes": l1,
        "l1_pct": 100 * l1 / arch["l1_timeloop_view_bytes"],
        "l1_ok": l1 <= arch["l1_timeloop_view_bytes"],
    }


def check_l2_capacity(k_inner: int, n_total: int, arch: dict) -> dict:
    """
    L2 共享容量校验（keep Weights；K_outer 已在 L2 temporal 拆分，所以单次只持 K_inner × N_total weights）。
    """
    l2 = k_inner * n_total * arch["fp16_bytes"]
    return {
        "l2_bytes": l2,
        "l2_pct": 100 * l2 / arch["l2_timeloop_view_bytes"],
        "l2_ok": l2 <= arch["l2_timeloop_view_bytes"],
    }


# ── Auto-derive 工具 ────────────────────────────────────────────────────
def derive_k_inner(K: int, m_cube: int, n_cube: int, arch: dict) -> int:
    """
    选择能整除 K 且满足 L0A/L0B 容量约束的最大 K_inner。
    L0A 约束: m_cube × k_inner × n_cube × 2 ≤ L0A
    L0B 约束: k_inner × n_cube × m_cube × 2 ≤ L0B
    """
    l0a_max = arch["l0a_bytes"] // (m_cube * n_cube * arch["fp16_bytes"])
    l0b_max = arch["l0b_bytes"] // (n_cube * m_cube * arch["fp16_bytes"])
    safe_max = min(l0a_max, l0b_max, K)

    # 寻找能整除 K 的最大值 ≤ safe_max
    for d in range(safe_max, 0, -1):
        if K % d == 0:
            return d
    raise ValueError(f"K={K} 没有可行的 k_inner ≤ {safe_max}")


def derive_m_l1(M_per_core: int, k_inner: int, n_cube: int, arch: dict) -> int:
    """
    选择能整除 M_per_core 且满足 L1 input 容量约束的最大 M_l1。
    L1 约束: m_l1 × k_inner × n_cube × 2 ≤ L1（Timeloop view）
    """
    l1_max = arch["l1_timeloop_view_bytes"] // (k_inner * n_cube * arch["fp16_bytes"])
    safe_max = min(l1_max, M_per_core)

    for d in range(safe_max, 0, -1):
        if M_per_core % d == 0:
            return d
    raise ValueError(f"M_per_core={M_per_core} 没有可行的 m_l1 ≤ {safe_max}")


# ── 主生成函数 ──────────────────────────────────────────────────────────
def generate_mapping(
    M: int, N: int, K: int,
    m_l2_spatial: int, n_l2_spatial: int,
    m_cube_spatial: int = None, n_cube_spatial: int = None,
    k_cube_spatial: int = None,
    arch: dict = ARCH_910B4,
) -> tuple[str, dict]:
    """
    自动派生切片 + 生成 Timeloop manual mapping yaml。

    Returns: (yaml_str, info_dict) where info_dict 含 derived 参数 + capacity 报告
    """
    if m_cube_spatial is None:
        m_cube_spatial = arch["cube_m"]
    if n_cube_spatial is None:
        n_cube_spatial = arch["cube_n"]
    if k_cube_spatial is None:
        k_cube_spatial = arch.get("cube_k", 1)   # L#1 fix: K-spatial 默认 16

    # 1. spatial 上限校验（含 K 维度，4096 = 16×16×16）
    if m_l2_spatial * n_l2_spatial > arch["n_cores"]:
        raise ValueError(f"L2 spatial {m_l2_spatial}×{n_l2_spatial} > n_cores={arch['n_cores']}")
    cube_total = m_cube_spatial * n_cube_spatial * k_cube_spatial
    if cube_total > arch["n_macs_per_core"]:
        raise ValueError(f"Cube spatial M×N×K = {cube_total} > n_macs_per_core={arch['n_macs_per_core']}")

    # 2. 整除性
    m_total_spatial = m_l2_spatial * m_cube_spatial
    n_total_spatial = n_l2_spatial * n_cube_spatial
    if M % m_total_spatial != 0:
        raise ValueError(f"M={M} 不能被 m_total_spatial={m_total_spatial} 整除")
    if N % n_total_spatial != 0:
        raise ValueError(f"N={N} 不能被 n_total_spatial={n_total_spatial} 整除")
    if K % k_cube_spatial != 0:
        raise ValueError(f"K={K} 不能被 k_cube_spatial={k_cube_spatial} 整除")

    M_per_core = M // m_l2_spatial
    N_per_core = N // n_l2_spatial
    K_after_cube_spatial = K // k_cube_spatial   # K 剩余 temporal 部分

    # 3. Auto-derive K_inner_temporal (Cube_Reg temporal K)
    # 容量约束（含 K-spatial 广播倍率）：
    #   L0A = m_cube × k_inner_temp × n_cube × k_cube_spatial × 2 ≤ L0A_cap
    #   L0B = k_inner_temp × n_cube × m_cube × k_cube_spatial × 2 ≤ L0B_cap
    fp16 = arch["fp16_bytes"]
    # Timeloop 看 L0A/L0B = 131072 bytes (depth × 4 cluster)，但实际硬件 65536，按 Timeloop 视角算
    l0_timeloop_view = 131072  # Timeloop 实际报告的 L0B/L0A capacity
    max_k_temp = l0_timeloop_view // (m_cube_spatial * n_cube_spatial * k_cube_spatial * fp16)
    safe_max_k_temp = min(max_k_temp, K_after_cube_spatial)
    if safe_max_k_temp < 1:
        raise ValueError(f"k_cube_spatial={k_cube_spatial} 太大 → L0 容量根本放不下 1 个 k_inner_temp")
    k_inner_temp_candidates = [d for d in range(safe_max_k_temp, 0, -1) if K_after_cube_spatial % d == 0]
    if not k_inner_temp_candidates:
        raise ValueError(f"K_after_cube_spatial={K_after_cube_spatial} 没有可行 k_inner_temp ≤ {safe_max_k_temp}")
    k_inner_temp = k_inner_temp_candidates[0]
    k_outer = K_after_cube_spatial // k_inner_temp
    k_inner = k_inner_temp * k_cube_spatial   # 总 K_inner = spatial × temporal

    # 4. Auto-derive M_l1（L1 input 容量约束下能整除 M_per_core 的最大）
    m_l1 = derive_m_l1(M_per_core, k_inner, n_cube_spatial, arch)
    m_l1_iter = M_per_core // m_l1     # L1 M 方向 iter 数

    # M 在 L1 内部还需要 cube spatial 拆分
    m_per_l1_inner = m_l1 // m_cube_spatial  # L0A temporal M
    if m_l1 % m_cube_spatial != 0:
        raise ValueError(f"m_l1={m_l1} 不能被 m_cube_spatial={m_cube_spatial} 整除")

    # N 在 per-core 内部：N_per_core / n_cube_spatial = N_l0c 总 temporal
    n_per_core_temporal = N_per_core // n_cube_spatial
    if N_per_core % n_cube_spatial != 0:
        raise ValueError(f"N_per_core={N_per_core} 不能被 n_cube_spatial={n_cube_spatial} 整除")

    # 5. capacity 校验（K-spatial 后 L0A/L0B 用 k_inner_temp，不是总 k_inner）
    cap_l0 = check_l0_capacity(m_cube_spatial, n_cube_spatial, k_inner_temp, arch)
    cap_l1 = check_l1_capacity(m_l1, k_inner_temp, n_cube_spatial, arch)
    cap_l2 = check_l2_capacity(k_inner, N, arch)

    errors = []
    for k_, name_ in [("l0a_ok", "L0A"), ("l0b_ok", "L0B"), ("l0c_ok", "L0C")]:
        if not cap_l0[k_]:
            errors.append(f"{name_} overflow: {cap_l0[name_.lower() + '_bytes']}")
    if not cap_l1["l1_ok"]:
        errors.append(f"L1 overflow: {cap_l1['l1_bytes']} > {arch['l1_timeloop_view_bytes']}")
    if not cap_l2["l2_ok"]:
        errors.append(f"L2 overflow: {cap_l2['l2_bytes']} > {arch['l2_timeloop_view_bytes']}")
    if errors:
        raise ValueError("Capacity check failed:\n  " + "\n  ".join(errors))

    info = {
        "m_l2_spatial": m_l2_spatial, "n_l2_spatial": n_l2_spatial,
        "m_cube_spatial": m_cube_spatial, "n_cube_spatial": n_cube_spatial,
        "M_per_core": M_per_core, "N_per_core": N_per_core,
        "k_inner": k_inner, "k_outer": k_outer,
        "m_l1": m_l1, "m_l1_iter": m_l1_iter,
        "m_per_l1_inner": m_per_l1_inner,
        "n_per_core_temporal": n_per_core_temporal,
        "cap_l0": cap_l0, "cap_l1": cap_l1, "cap_l2": cap_l2,
    }

    # 6. 生成 yaml
    # 维度分布:
    #   M = m_l2_spatial × m_cube_spatial × m_per_l1_inner × m_l1_iter
    #   N = n_l2_spatial × n_cube_spatial × n_per_core_temporal (放 L0C 或 L1)
    #   K = k_outer (at L2) × k_inner (at Cube_Reg)
    #
    # 默认把 N temporal 放在 L0C，把 M_l1_iter 放 L1 temporal:
    # - Cube_Reg temporal: K=k_inner
    # - L0A temporal: M=m_per_l1_inner（L0A 视角下 M 时序循环）
    # - L0B temporal: 1 (M N K 都是 1，weights 直接来 L2)
    # - L0C spatial: M=m_cube N=n_cube K=1 split=3
    # - L0C temporal: N=n_per_core_temporal
    # - L1 temporal: M=m_l1_iter
    # - L2 spatial: M=m_l2 N=n_l2 K=1 split=3
    # - L2 temporal: K=k_outer
    # - DRAM: 1

    yaml_str = f"""# 手写 mapping（generate_manual_mapping.py v0.2 自动生成）
# ──────────────────────────────────────────────────────────────────────────
# 工作负载: M={M} N={N} K={K}
# 架构: {arch['name']} ({arch['n_cores']} AIC × {arch['n_macs_per_core']} MAC/AIC)
#
# Spatial 分解:
#   L2_Buffer  spatial: M={m_l2_spatial} N={n_l2_spatial} K=1  ({m_l2_spatial*n_l2_spatial}/{arch['n_cores']} cores)
#   L0C_Buffer spatial: M={m_cube_spatial} N={n_cube_spatial} K={k_cube_spatial}  ({m_cube_spatial*n_cube_spatial*k_cube_spatial}/{arch['n_macs_per_core']} MACs/AIC)
#
# Temporal 分解:
#   Cube_Reg   temporal: K={k_inner}        (L0 K reduce; cube 内时序)
#   L0A_Buffer temporal: M={m_per_l1_inner} (L1 内 M iter)
#   L0B_Buffer temporal: (1)                (Weights 不在 L0B 时序)
#   L0C_Buffer temporal: N={n_per_core_temporal}  (L1 内 N iter, 输出累加)
#   L1_Buffer  temporal: M={m_l1_iter}      (L2→L1 M 切片 iter)
#   L2_Buffer  temporal: (1)                (K_outer 放 DRAM 让 L2 只持 K_inner slice)
#   DRAM       temporal: K={k_outer}        (DRAM→L2 K_outer 次重载 weight tile)
#
# 维度乘积:
#   M = {m_l2_spatial} × {m_cube_spatial} × {m_per_l1_inner} × {m_l1_iter} = {m_l2_spatial * m_cube_spatial * m_per_l1_inner * m_l1_iter} (target {M}) {'✓' if m_l2_spatial * m_cube_spatial * m_per_l1_inner * m_l1_iter == M else '✗'}
#   N = {n_l2_spatial} × {n_cube_spatial} × {n_per_core_temporal} = {n_l2_spatial * n_cube_spatial * n_per_core_temporal} (target {N}) {'✓' if n_l2_spatial * n_cube_spatial * n_per_core_temporal == N else '✗'}
#   K = {k_outer} × {k_inner} = {k_outer * k_inner} (target {K}) {'✓' if k_outer * k_inner == K else '✗'}
#
# Capacity check (Timeloop view):
#   L0A: {cap_l0['l0a_bytes']:>7} bytes ({cap_l0['l0a_pct']:5.1f}% of 64 KB)
#   L0B: {cap_l0['l0b_bytes']:>7} bytes ({cap_l0['l0b_pct']:5.1f}% of 64 KB)
#   L0C: {cap_l0['l0c_bytes']:>7} bytes ({cap_l0['l0c_pct']:5.1f}% of 128 KB)
#   L1:  {cap_l1['l1_bytes']:>7} bytes ({cap_l1['l1_pct']:5.1f}% of {arch['l1_timeloop_view_bytes']//1024} KB Timeloop view)
#   L2:  {cap_l2['l2_bytes']:>7} bytes ({cap_l2['l2_pct']:5.1f}% of {arch['l2_timeloop_view_bytes']//1024//1024} MB Timeloop view, **L#2 fix**)

mapping:

  # ── Datatype keep/bypass（架构-correct dataflow）─────────────────────
  - target: Cube_Reg
    type: datatype
    keep: []
    bypass: [Weights, Inputs, Outputs]

  - target: L0C_Buffer
    type: datatype
    keep: [Outputs]
    bypass: [Weights, Inputs]

  - target: L0A_Buffer
    type: datatype
    keep: [Inputs]
    bypass: [Weights, Outputs]

  - target: L0B_Buffer
    type: datatype
    keep: [Weights]
    bypass: [Inputs, Outputs]

  - target: L1_Buffer
    type: datatype
    keep: [Inputs]
    bypass: [Weights, Outputs]

  - target: L2_Buffer
    type: datatype
    keep: [Weights]
    bypass: [Inputs, Outputs]

  - target: DRAM
    type: datatype
    keep: [Weights, Inputs, Outputs]
    bypass: []

  # ── 时空映射（innermost → outermost）──────────────────────────────────

  # Cube_Reg: 内层 K reduce temporal (k_inner_temp cycles, 配合 k_cube_spatial=16 spatial)
  - target: Cube_Reg
    type: temporal
    factors: M1 N1 K{k_inner_temp}
    permutation: KMN

  # L0A: M temporal 切片（per-core M_per_l1_inner 循环）
  - target: L0A_Buffer
    type: temporal
    factors: M{m_per_l1_inner} N1 K1
    permutation: MNK

  # L0B: 不切片（每次 L2→L0B 重载 K_inner × N_inner tile）
  - target: L0B_Buffer
    type: temporal
    factors: M1 N1 K1
    permutation: MNK

  # L0C spatial: 16×16×16 Cube fanout (4096 MAC 全占用) — split=3 (1D arch)
  - target: L0C_Buffer
    type: spatial
    factors: M{m_cube_spatial} N{n_cube_spatial} K{k_cube_spatial}
    permutation: MNK
    split: 3

  # L0C temporal: N 时序累加（输出 partial sum 在 L0C）
  - target: L0C_Buffer
    type: temporal
    factors: M1 N{n_per_core_temporal} K1
    permutation: NMK

  # L1: M_l1 切片 iter（L2 → L1 加载新 M tile）
  - target: L1_Buffer
    type: temporal
    factors: M{m_l1_iter} N1 K1
    permutation: MNK

  # L2 spatial: 24-core fanout（split=3）
  - target: L2_Buffer
    type: spatial
    factors: M{m_l2_spatial} N{n_l2_spatial} K1
    permutation: MNK
    split: 3

  # L2 temporal: 不切片（K_outer 放 DRAM 让 L2 只持 K_inner slice）
  - target: L2_Buffer
    type: temporal
    factors: M1 N1 K1
    permutation: MNK

  # DRAM temporal: K_outer 重载 weight tile 到 L2
  - target: DRAM
    type: temporal
    factors: M1 N1 K{k_outer}
    permutation: KMN
"""
    return yaml_str, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--m-l2-spatial", type=int, default=4)
    p.add_argument("--n-l2-spatial", type=int, default=6)
    p.add_argument("--m-cube-spatial", type=int, default=16)
    p.add_argument("--n-cube-spatial", type=int, default=16)
    p.add_argument("--output", type=str, default="-")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    try:
        yaml_str, info = generate_mapping(
            M=args.M, N=args.N, K=args.K,
            m_l2_spatial=args.m_l2_spatial, n_l2_spatial=args.n_l2_spatial,
            m_cube_spatial=args.m_cube_spatial, n_cube_spatial=args.n_cube_spatial,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"[info] M_per_core={info['M_per_core']} N_per_core={info['N_per_core']}", file=sys.stderr)
        print(f"[info] k_inner={info['k_inner']} k_outer={info['k_outer']}", file=sys.stderr)
        print(f"[info] m_l1={info['m_l1']} m_l1_iter={info['m_l1_iter']} m_per_l1_inner={info['m_per_l1_inner']}", file=sys.stderr)
        print(f"[info] L0A {info['cap_l0']['l0a_pct']:.1f}% / L0B {info['cap_l0']['l0b_pct']:.1f}% / L0C {info['cap_l0']['l0c_pct']:.1f}% / L1 {info['cap_l1']['l1_pct']:.1f}% / L2 {info['cap_l2']['l2_pct']:.1f}%", file=sys.stderr)

    if args.output == "-":
        print(yaml_str)
    else:
        Path(args.output).write_text(yaml_str, encoding="utf-8")
        print(f"OK  {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
