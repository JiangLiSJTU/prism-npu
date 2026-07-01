#!/usr/bin/env python3
"""
Phase J — 910B4-anchored 架构 sweep v3（Phase N N6b：pipe-aware 公式）

核心区别 vs v2（仅 Cube + Vector ALU）：
  v3 把 wall-clock 拆成 7 类 pipe + host gap：
    - aic_pipes:  mac, mte1 (L1↔L0), mte2 (HBM↔L1), fixpipe (L0C→L1/UB), scalar
    - aiv_pipes:  vec (ALU), mte2 (UB↔L1), mte3 (UB→output), scalar, idle
    - host_gap:   per-kernel CANN runtime 开销（arch-invariant）

  每个 pipe 独立 scaling（按 arch 资源比例），然后取 max 作为 critical path：
    aic_time_new = aic_bubble_baseline + max(scaled_aic_pipes)
    aiv_time_new = aiv_idle_baseline + max(scaled_aiv_pipes)
    wall_clock = aic_time_new + aiv_time_new + host_gap_new

数据源：
  - data/pipe_baseline_per_model.json：9 配置 PipeUtil 实测 + 2 占位/继承
  - arch/ascend_910b4_for_sweep_v2.yaml：baseline arch + 12 个细粒度字段

输出：
  - data/phase_j_sweep_v3.json：12 维 sweep × 7 模型 × ratio + pipe breakdown

参考：
  docs/overhead_decomposition_audit.md (v1.1)
  docs/architecture_sweep_report.md (v2 旧版)
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent


# ─────────────────────────────────────────────────────────────────────
# Baseline 910B4 arch（从 yaml 复制；如未来 yaml 改变 sync 即可）
# ─────────────────────────────────────────────────────────────────────
BASELINE_910B4 = {
    'name':                   '910B4-baseline',
    'n_cores':                24,
    'cube_m':                 16,  'cube_n': 16,  'cube_k': 16,
    'cube_total_macs':        24 * 16 * 16 * 16,   # 98,304 MAC/cycle 全 chip
    'l2_mb':                  96,
    'hbm_bw_gbs':             392,
    'l1_kb':                  512,
    'l0a_kb':                 64,  'l0b_kb': 64,  'l0c_kb': 128,
    'l1_l0_bw_gbs':           2048,
    'fixpipe_bw_gbs':         4096,
    'aiv_per_aic':            2,
    'ub_kb_per_aiv':          192,
    'ub_l1_bw_gbs':           2048,
    'aiv_lanes_per_aiv':      128,
    'aiv_total_throughput':   24 * 2 * 128,  # 6144 OP/cycle 全 chip
    'tdp_w':                  300,
    'clock_ghz':              1.6,
    'fp16_tflops':            280,
    'beta_host_gap_us_per_kernel': 41.6,
    'ub_l1_fused':            False,    # 默认不融合
}


# ─────────────────────────────────────────────────────────────────────
# Pipe baseline（per-model 实测数据）
# ─────────────────────────────────────────────────────────────────────
def load_pipe_baseline(path: Path) -> dict:
    """加载 per-model pipe breakdown 实测数据（msprof PipeUtilization）"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data['configs']


# aic_fixpipe / aiv_mte3 的 gm_frac 缺省值（用于不在 pipe_dest_bw.json 中的
# config，例如 predict_pipe 合成 baseline）。0.5 = "未知，假设半数字节写 GM"。
DEFAULT_FIXPIPE_GM_FRAC = 0.5
DEFAULT_AIV_MTE3_GM_FRAC = 0.5


def _dest_time_proxy(gm_frac: float, arch: dict, onchip_bw_key: str,
                     fused_eliminates: bool = False) -> float:
    """目的地相关 pipe（aic_fixpipe / aiv_mte3）的有效搬运耗时代理（∝ 1/有效带宽）。

    这两条 pipe 的 store 分两个目的地、带宽差 ~5-10×（见 Issue #7）：
      → GM   (gm_frac 字节)      → hbm_bw
      → 片上 (1-gm_frac 字节)    → onchip_bw_key 指定的带宽（fixpipe_bw / ub_l1_bw）
    fused_eliminates=True 时 UB+L1 融合近乎消除片上通路（5% 残余）——仅 aiv_mte3 适用。
    """
    onchip = (1.0 - gm_frac) / max(arch[onchip_bw_key], 1e-9)
    if fused_eliminates and arch.get('ub_l1_fused', False):
        onchip *= 0.05
    gm = gm_frac / max(arch['hbm_bw_gbs'], 1e-9)
    return gm + onchip


def _dest_blend_factor(gm_frac: float, baseline_arch: dict, variant_arch: dict,
                       onchip_bw_key: str, fused_eliminates: bool = False) -> float:
    """variant/baseline 的有效带宽缩放因子（逐 variant 重算 hbm/片上 blend）。"""
    return _dest_time_proxy(gm_frac, variant_arch, onchip_bw_key, fused_eliminates) \
        / max(_dest_time_proxy(gm_frac, baseline_arch, onchip_bw_key, fused_eliminates), 1e-12)


# ─────────────────────────────────────────────────────────────────────
# AIC pipe 物理 scaling
# ─────────────────────────────────────────────────────────────────────
def scale_aic_pipes(pipes_baseline: dict, baseline_arch: dict, variant_arch: dict,
                    fixpipe_gm_frac: float = DEFAULT_FIXPIPE_GM_FRAC) -> dict:
    """各 AIC pipe 按 arch 资源比例缩放。

    每个 pipe time 假设与对应的 arch 资源 throughput 成反比：
      pipe_time_new = pipe_time_baseline × (resource_baseline / resource_new)

    pipe → resource:
      mac     → cube_total_macs × clock
      mte1    → l1_l0_bw × clock
      mte2    → hbm_bw × clock      (实际是 hbm_bw × prefetch；clock 基本不影响 BW)
      fixpipe → blend(hbm_bw, fixpipe_bw)  按 per-config gm_frac —— FixPipe 的
                L0C→输出 store 分 L0C→GM 直写（hbm_bw）与 L0C→L1/UB（fixpipe_bw）；
                OLS 实测多数 config gm_frac 0.4-1.0（见 Issue #7 / pipe_dest_bw.json）
      scalar  → arch-invariant（小 op 控制流）
    """
    cube_throughput_baseline = baseline_arch['cube_total_macs'] * baseline_arch['clock_ghz']
    cube_throughput_variant  = variant_arch['cube_total_macs']  * variant_arch['clock_ghz']
    l1_l0_bw_baseline = baseline_arch['l1_l0_bw_gbs']
    l1_l0_bw_variant  = variant_arch['l1_l0_bw_gbs']
    hbm_bw_baseline = baseline_arch['hbm_bw_gbs']
    hbm_bw_variant  = variant_arch['hbm_bw_gbs']

    # fixpipe：有效带宽 = hbm_bw 与 fixpipe_bw 的 blend（按 gm_frac），逐 variant 重算
    fixpipe_factor = _dest_blend_factor(fixpipe_gm_frac, baseline_arch, variant_arch,
                                        'fixpipe_bw_gbs')

    return {
        'mac':     pipes_baseline['mac']     * (cube_throughput_baseline / max(cube_throughput_variant, 1e-9)),
        'mte1':    pipes_baseline['mte1']    * (l1_l0_bw_baseline / max(l1_l0_bw_variant, 1e-9)),
        'mte2':    pipes_baseline['mte2']    * (hbm_bw_baseline / max(hbm_bw_variant, 1e-9)),
        'fixpipe': pipes_baseline['fixpipe'] * fixpipe_factor,
        'scalar':  pipes_baseline['scalar'],   # arch-invariant
    }


# ─────────────────────────────────────────────────────────────────────
# AIV pipe 物理 scaling
# ─────────────────────────────────────────────────────────────────────
def scale_aiv_pipes(pipes_baseline: dict, baseline_arch: dict, variant_arch: dict,
                    mte3_gm_frac: float = DEFAULT_AIV_MTE3_GM_FRAC) -> dict:
    """AIV pipes 按对应 arch 资源缩放。

    pipe → resource:
      vec    → aiv_total_throughput × clock
      mte2   → ub_l1_bw × clock     (UB+L1 fused → 实际等价于内部带宽 ↑↑↑，等同消除)
      mte3   → blend(hbm_bw, ub_l1_bw)  按 per-config gm_frac 加权 —— MTE3 的
               store 分 UB→GM（hbm_bw）与 UB→L1（ub_l1_bw）；gm_frac OLS 实测自
               data/calibration/pipe_dest_bw.json（见 08_predict_pipe.md §3.6）
      scalar → arch-invariant
      idle   → arch-invariant（已含 aic-aiv 同步等待，新公式假设不变）
    """
    aiv_throughput_baseline = baseline_arch['aiv_total_throughput'] * baseline_arch['clock_ghz']
    aiv_throughput_variant  = variant_arch['aiv_total_throughput']  * variant_arch['clock_ghz']
    ub_l1_bw_baseline = baseline_arch['ub_l1_bw_gbs']
    ub_l1_bw_variant  = variant_arch['ub_l1_bw_gbs']

    # UB+L1 融合假设：aiv_mte2 时间 → 0（fused 内存池消除 UB↔L1 流量）
    # 保留 5% 残余作为 microarch 控制流近似
    if variant_arch.get('ub_l1_fused', False):
        mte2_factor = 0.05
    else:
        mte2_factor = ub_l1_bw_baseline / max(ub_l1_bw_variant, 1e-9)

    # mte3：有效带宽 = hbm_bw 与 ub_l1_bw 的 blend（按 gm_frac），逐 variant 重算；
    # UB+L1 融合消除片上分量（fused_eliminates=True）
    mte3_factor = _dest_blend_factor(mte3_gm_frac, baseline_arch, variant_arch,
                                     'ub_l1_bw_gbs', fused_eliminates=True)

    return {
        'vec':    pipes_baseline['vec']    * (aiv_throughput_baseline / max(aiv_throughput_variant, 1e-9)),
        'mte2':   pipes_baseline['mte2']   * mte2_factor,
        'mte3':   pipes_baseline['mte3']   * mte3_factor,
        'scalar': pipes_baseline['scalar'],
        'idle':   pipes_baseline['idle'],
    }


# ─────────────────────────────────────────────────────────────────────
# Predict wall-clock v3
# ─────────────────────────────────────────────────────────────────────
def predict_wallclock_v3(model_pipe: dict, variant_arch: dict, baseline_arch: dict = BASELINE_910B4) -> dict:
    """v3 pipe-aware wall-clock 预测。

    Args:
        model_pipe:    per-inference pipe baseline (data/pipe_baseline_per_model.json[cfg])
        variant_arch:  目标 arch 字典
        baseline_arch: 锚点 arch（默认 910B4）

    Returns:
        dict 含 wall_clock_us, aic_time_us, aiv_time_us, host_gap_us, per-pipe breakdown
    """
    # 1) AIC pipe scaling
    aic_pipes_baseline = model_pipe['aic_pipes_us']
    fixpipe_gm_frac = model_pipe.get('_aic_fixpipe_gm_frac', DEFAULT_FIXPIPE_GM_FRAC)
    aic_pipes_new = scale_aic_pipes(aic_pipes_baseline, baseline_arch, variant_arch, fixpipe_gm_frac)

    # AIC bubble (假设 arch-invariant；实际 arch 改 bubble 也会变，但一阶近似可接受)
    aic_bubble = model_pipe.get('aic_bubble_us', 0)
    aic_max_new = max(aic_pipes_new.values())
    aic_time_new = aic_max_new + aic_bubble

    # 2) AIV pipe scaling
    # 关键：AIV 的 'idle' 不参与 max() 计算（它是 bubble 不是 active pipe）
    # baseline 模型上，aiv_time = max(active_pipes) + aiv_bubble，aiv_bubble 含 idle + AIC 同步等待
    # 所以预测时：aiv_time_new = max(scaled_active_pipes) + aiv_bubble_baseline
    aiv_pipes_baseline = model_pipe['aiv_pipes_us']
    mte3_gm_frac = model_pipe.get('_aiv_mte3_gm_frac', DEFAULT_AIV_MTE3_GM_FRAC)
    aiv_pipes_new_all = scale_aiv_pipes(aiv_pipes_baseline, baseline_arch, variant_arch, mte3_gm_frac)
    aiv_pipes_new = {k: v for k, v in aiv_pipes_new_all.items() if k != 'idle'}
    aiv_active_max_new      = max(aiv_pipes_new.values())
    aiv_active_max_baseline = max((v for k, v in aiv_pipes_baseline.items() if k != 'idle'), default=0)
    aiv_time_baseline       = model_pipe['aiv_time_us']
    aiv_bubble = max(0.0, aiv_time_baseline - aiv_active_max_baseline)
    aiv_time_new = aiv_active_max_new + aiv_bubble

    # 3) Kernel-internal gap (AIC/AIV 同步等待，arch-invariant 一阶近似)
    # 来自 task_dur - (aic + aiv)；负值意味着 baseline 上 AIC/AIV 重叠，clamp 到 0
    kernel_gap_baseline = max(0.0, model_pipe.get('kernel_gap_us', 0))

    # 4) Host gap：每 kernel 的 host overhead × n_kernels
    n_kernels = model_pipe['n_kernels_per_inf']
    host_gap_us_per_kernel = model_pipe.get('host_gap_us_per_kernel', baseline_arch['beta_host_gap_us_per_kernel'])
    # arch 影响 host gap：不影响（host_gap 是 software-only）
    host_gap_new = n_kernels * host_gap_us_per_kernel

    # 5) wall_clock = AIC + AIV + kernel_gap + host_gap (serial 假设)
    wall_clock = aic_time_new + aiv_time_new + kernel_gap_baseline + host_gap_new

    return {
        'aic_time_us':        round(aic_time_new, 2),
        'aiv_time_us':        round(aiv_time_new, 2),
        'kernel_gap_us':      round(kernel_gap_baseline, 2),
        'host_gap_us':        round(host_gap_new, 2),
        'wall_clock_us':      round(wall_clock, 2),
        'aic_pipes_scaled':   {k: round(v, 2) for k, v in aic_pipes_new.items()},
        'aiv_pipes_scaled':   {k: round(v, 2) for k, v in aiv_pipes_new.items()},
        'aic_dominant_pipe':  max(aic_pipes_new, key=aic_pipes_new.get),
        'aiv_dominant_pipe':  max(aiv_pipes_new, key=aiv_pipes_new.get),
    }


# ─────────────────────────────────────────────────────────────────────
# Sweep 维度定义
# ─────────────────────────────────────────────────────────────────────
SWEEP = {
    'n_cores':              [8, 12, 16, 24, 32, 48],
    'cube_kdim':            [(8,8,16), (8,16,16), (16,16,16), (16,16,32), (32,32,16)],
    'l2_mb':                [8, 16, 32, 96, 192, 384],
    'hbm_bw_gbs':           [50, 100, 392, 800],
    'aiv_per_aic':          [1, 2, 4],
    'tdp_w':                [100, 150, 200, 300, 400],
    'l0a_kb':               [32, 64, 128, 256],
    'l1_kb':                [256, 512, 1024],
    'l1_l0_bw_gbs':         [1024, 2048, 4096, 8192],
    'fixpipe_bw_gbs':       [2048, 4096, 8192, 16384],
    'ub_l1_fused':          [False, True],
    # `beta_host_gap_us_per_kernel` 不在 sweep 维度中 —— host_gap 视为 software-only /
    # arch-invariant（见 predict_wallclock_v3 注释 "arch 影响 host gap：不影响"）。
    # CANN runtime 优化的 host_gap 杠杆由 prism-ceiling 的 S2 情景建模，不归架构 sweep。
    # 历史：Issue #8 之前曾把 [10, 41.6, 100] 列为 sweep 维度但 predict_wallclock_v3
    # 不消费 variant_arch[...] —— 3 个变体产出完全相同结果，已移除。
    # `BASELINE_910B4['beta_host_gap_us_per_kernel']` 保留作 fallback（model_pipe 无
    # 实测 host_gap_us_per_kernel 时用）。
}


def make_arch_variant(dim: str, value, baseline=BASELINE_910B4) -> dict:
    """生成单维度 arch variant"""
    arch = deepcopy(baseline)
    arch['variant_dim'] = dim
    arch['variant_value'] = str(value)

    if dim == 'cube_kdim':
        arch['cube_m'], arch['cube_n'], arch['cube_k'] = value
        arch['cube_total_macs'] = arch['n_cores'] * value[0] * value[1] * value[2]
        arch['name'] = f"cube{value[0]}x{value[1]}x{value[2]}"
    elif dim == 'tdp_w':
        # TDP 缩放代理：clock ∝ TDP^(1/3)
        ratio = value / baseline['tdp_w']
        arch['clock_ghz'] = baseline['clock_ghz'] * (ratio ** (1/3))
        arch['tdp_w'] = value
        arch['name'] = f"TDP{value}W"
    elif dim == 'n_cores':
        arch['n_cores'] = value
        arch['cube_total_macs'] = value * arch['cube_m'] * arch['cube_n'] * arch['cube_k']
        arch['aiv_total_throughput'] = value * arch['aiv_per_aic'] * arch['aiv_lanes_per_aiv']
        arch['name'] = f"n_cores={value}"
    elif dim == 'aiv_per_aic':
        arch['aiv_per_aic'] = value
        arch['aiv_total_throughput'] = arch['n_cores'] * value * arch['aiv_lanes_per_aiv']
        arch['name'] = f"aiv_per_aic={value}"
    elif dim == 'ub_l1_fused':
        arch['ub_l1_fused'] = value
        arch['name'] = f"ub_l1_fused={value}"
    else:
        arch[dim] = value
        arch['name'] = f"{dim}={value}"

    return arch


# ─────────────────────────────────────────────────────────────────────
# TCO proxy（含 v3 新维度的 die area 估算）
# ─────────────────────────────────────────────────────────────────────
def estimate_tco(arch: dict) -> dict:
    cube_macs_per_core = arch['cube_m'] * arch['cube_n'] * arch['cube_k']
    die_area = (
        arch['n_cores'] * cube_macs_per_core / 1e3
        + arch['l2_mb'] * 0.5
        + arch['n_cores'] * arch['aiv_per_aic'] * arch['aiv_lanes_per_aiv'] / 100
        + arch['n_cores'] * arch['l1_kb'] / 1024 * 0.5
        + arch['n_cores'] * arch['aiv_per_aic'] * arch['ub_kb_per_aiv'] / 1024 * 0.5
    )
    # FixPipe / L1↔L0 BW 提升的 die 代价（连线密度 + buffer）
    die_area += (arch['l1_l0_bw_gbs'] / 2048 - 1) * 5    # 每 +1024 GB/s ≈ +5 mm²
    die_area += (arch['fixpipe_bw_gbs'] / 4096 - 1) * 5
    if arch.get('ub_l1_fused', False):
        die_area += 8   # 融合需要重设计内存层级，~8 mm² 估值

    if arch['hbm_bw_gbs'] >= 392:
        mem_cost = arch['hbm_bw_gbs'] / 100
    else:
        mem_cost = arch['hbm_bw_gbs'] / 200

    return {
        'die_area_proxy': round(die_area, 1),
        'power_proxy':    arch['tdp_w'],
        'mem_cost_proxy': round(mem_cost, 2),
        'tco_score':      round(die_area * 0.4 + arch['tdp_w'] * 0.3 + mem_cost * 100 * 0.3, 1),
    }


# ─────────────────────────────────────────────────────────────────────
# 验证：baseline arch 上 v3 公式输出 vs 实测 wall_clock
# ─────────────────────────────────────────────────────────────────────
def verify_baseline(pipe_baseline: dict) -> dict:
    """跑 baseline arch 上每个 model 的 v3 公式预测，对比实测 wall_clock_us。"""
    results = {}
    print(f"\n{'config':<32}{'wall_pred':>12}{'wall_meas':>12}{'err%':>8}{'aic':>8}{'aiv':>8}{'host':>9}")
    for cfg, model_pipe in pipe_baseline.items():
        pred = predict_wallclock_v3(model_pipe, BASELINE_910B4, BASELINE_910B4)
        meas = model_pipe['wall_clock_us']
        err_pct = 100 * (pred['wall_clock_us'] - meas) / meas if meas > 0 else 0
        results[cfg] = {
            'wall_pred_us': pred['wall_clock_us'],
            'wall_meas_us': meas,
            'err_pct':      round(err_pct, 2),
            'aic_pred_us':  pred['aic_time_us'],
            'aiv_pred_us':  pred['aiv_time_us'],
            'host_gap_us':  pred['host_gap_us'],
        }
        print(f"{cfg:<32}{pred['wall_clock_us']:>12.0f}{meas:>12.0f}{err_pct:>7.1f}%"
              f"{pred['aic_time_us']:>8.0f}{pred['aiv_time_us']:>8.0f}{pred['host_gap_us']:>9.0f}")
    return results


# ─────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipe-baseline', default=str(REPO / 'data' / 'calibration' / 'pipe_baseline_per_model.json'))
    p.add_argument('--output',         default=str(REPO / 'data' / 'outputs' / 'phase_j_sweep_v3.json'))
    p.add_argument('--test-models', default=None,
                   help='逗号分隔的模型 key 列表；默认使用 pipe_baseline 的全部 config key。'
                        '跨项目复用时无需改名即可识别自有 baseline 的模型。')
    p.add_argument('--pipe-dest-bw',
                   default=str(REPO / 'data' / 'calibration' / 'pipe_dest_bw.json'),
                   help='aic_fixpipe / aiv_mte3 的 per-config gm_frac 校准文件'
                        '（UB/L0C→GM 字节占比，OLS 实测；见 Issue #7）。')
    args = p.parse_args()

    with open(args.pipe_baseline, encoding='utf-8') as f:
        baseline_data = json.load(f)
    pipe_baseline = baseline_data['configs']

    # 注入 aic_fixpipe / aiv_mte3 的 per-config gm_frac（→GM 字节占比）—— scale_*_pipes
    # 据此把这两条 pipe 在 hbm_bw 与片上带宽之间 blend 缩放。缺校准的 config 用缺省值。
    try:
        with open(args.pipe_dest_bw, encoding='utf-8') as f:
            dest_bw = json.load(f)
    except FileNotFoundError:
        dest_bw = {}
        print(f"  ⚠ 未找到 {args.pipe_dest_bw}，gm_frac 全部用缺省值")
    fixpipe_cal = dest_bw.get('aic_fixpipe', {})
    mte3_cal = dest_bw.get('aiv_mte3', {})

    def _gm(cal: dict, cfg: str, default: float) -> float:
        g = cal.get(cfg, {}).get('gm_frac')
        return g if isinstance(g, (int, float)) else default

    n_miss = 0
    for cfg, pipe in pipe_baseline.items():
        pipe['_aic_fixpipe_gm_frac'] = _gm(fixpipe_cal, cfg, DEFAULT_FIXPIPE_GM_FRAC)
        pipe['_aiv_mte3_gm_frac'] = _gm(mte3_cal, cfg, DEFAULT_AIV_MTE3_GM_FRAC)
        if cfg not in mte3_cal:
            n_miss += 1
    if n_miss:
        print(f"  ⚠ {n_miss} 个 config 不在 pipe-dest-bw 校准中，用缺省 gm_frac")

    # 1) baseline 验证：error < 5%（除 NetTrans 占位与继承的 Embedding）
    print("=" * 80)
    print("Step 1: baseline arch 验证 (期望 err% < 5% for measured configs)")
    print("=" * 80)
    verification = verify_baseline(pipe_baseline)

    # 2) 主 sweep
    print("\n" + "=" * 80)
    print("Step 2: 多维 sweep")
    print("=" * 80)

    if args.test_models:
        test_models = [m.strip() for m in args.test_models.split(',') if m.strip()]
        missing = [m for m in test_models if m not in pipe_baseline]
        if missing:
            print(f"  ⚠ --test-models 中 {len(missing)} 个 key 不在 pipe_baseline: {missing}")
    else:
        # 默认遍历 baseline 全部 config —— 跨项目复用时自动识别下游自有模型名。
        test_models = list(pipe_baseline.keys())

    baseline_results = {}
    print(f"\n=== Baseline 910B4 v3（per-inference μs）===")
    for m in test_models:
        if m not in pipe_baseline:
            continue
        r = predict_wallclock_v3(pipe_baseline[m], BASELINE_910B4, BASELINE_910B4)
        baseline_results[m] = r
        print(f"  {m:<32} wall={r['wall_clock_us']:>10.0f}  aic={r['aic_time_us']:>8.0f}  "
              f"aiv={r['aiv_time_us']:>8.0f}  host={r['host_gap_us']:>8.0f}")

    all_variants = []
    for dim, values in SWEEP.items():
        for v in values:
            arch = make_arch_variant(dim, v)
            tco = estimate_tco(arch)
            entry = {
                'variant':       arch['name'],
                'variant_dim':   dim,
                'variant_value': str(v),
                'arch':          {k: arch[k] for k in arch if k not in ('name', 'variant_dim', 'variant_value')
                                  and not callable(arch[k])},
                'tco':           tco,
                'models':        {},
            }
            for m in test_models:
                if m not in pipe_baseline:
                    continue
                r = predict_wallclock_v3(pipe_baseline[m], arch, BASELINE_910B4)
                ratio = r['wall_clock_us'] / max(baseline_results[m]['wall_clock_us'], 1e-9)
                entry['models'][m] = {**r, 'ratio_vs_baseline': round(ratio, 4)}
            all_variants.append(entry)

    # 3) 输出
    out = {
        'version':            'v3_pipe_aware',
        'baseline_arch':      BASELINE_910B4,
        'baseline_results':   baseline_results,
        'verification':       verification,
        'sweep_dimensions':   list(SWEEP.keys()),
        'n_variants':         len(all_variants),
        'variants':           all_variants,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n=> 写入 {args.output}（{len(all_variants)} 个变体）")

    # 4) 关键 ratio 总结（列按 test_models 动态生成）
    print("\n=== 关键 ratio 总结（每列一个 model，ratio_vs_baseline）===")
    col_models = [m for m in test_models if m in pipe_baseline]

    def fmt_ratio(models_dict: dict, key: str) -> str:
        if key not in models_dict:
            return '—'
        return f"{models_dict[key]['ratio_vs_baseline']:.2f}"

    print(f"{'variant':<26} | " + " ".join(f"{m[-9:]:>9}" for m in col_models))
    for entry in all_variants:
        cells = " ".join(f"{fmt_ratio(entry['models'], k):>9}" for k in col_models)
        print(f"{entry['variant']:<26} | {cells}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
