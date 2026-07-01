#!/usr/bin/env python3
"""
Phase M：把 msprof 实测每个 (M, N, K) 对照到 Timeloop manual mapping 预测。

步骤：
  1. 读 cube_util_extracted.json 的 top_shapes（按 aicore_time 排序的算子）
  2. 对每个 (M, N, K)，用 generate_manual_mapping + run_manual_mapping 跑 Timeloop
  3. 对比：cube_util_real (msprof) vs cycles_timeloop_theoretical
  4. 推导 η_real(shape) = aic_mac_fp16_ratio (直接从 msprof 来)
  5. 验证：real_wall_clock = cycles_timeloop / η_real / clock_hz × 1e6 应 ≈ aicore_time

输入：data/cube_util_extracted.json
输出：data/timeloop_vs_real_calibration.json + 终端汇总表
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent  # sim-experiment/
RUN_SCRIPT = REPO / "scripts" / "mapper" / "run_manual_mapping.py"
ARCH_YAML = REPO / "arch" / "ascend_910b4_for_mapping.yaml"   # L#2 派生 yaml
CLOCK_MHZ_910B4 = 1600   # 1.6 GHz


def find_l2_spatial(M, N, n_cores_max=24):
    """
    自动派生 L2 spatial (M_l2, N_l2)，使：
      - M_l2 × 16 | M（cube spatial M=16）
      - N_l2 × 16 | N（cube spatial N=16）
      - M_l2 × N_l2 ≤ n_cores_max（默认 24）
    最大化 M_l2 × N_l2 利用核数。
    """
    # M_l2 候选：M / 16 的所有因子
    if M < 16 or N < 16:
        return None, None
    m_per_core_max = M // 16   # M_l2 最大 = M / 16
    n_per_core_max = N // 16

    best_m_l2, best_n_l2, best_cores = 1, 1, 1
    for m_l2 in range(1, m_per_core_max + 1):
        if M % (m_l2 * 16) != 0:
            continue
        for n_l2 in range(1, n_per_core_max + 1):
            if N % (n_l2 * 16) != 0:
                continue
            cores = m_l2 * n_l2
            if cores > n_cores_max:
                continue
            if cores > best_cores:
                best_m_l2, best_n_l2, best_cores = m_l2, n_l2, cores

    return best_m_l2, best_n_l2


def run_timeloop_for_shape(M, N, K, name="auto"):
    """跑 Timeloop manual mapping，返回 cycles。"""
    m_l2, n_l2 = find_l2_spatial(M, N)
    if m_l2 is None or (m_l2 == 1 and n_l2 == 1):
        return None, f"No valid spatial decomposition for M={M} N={N}"

    cmd = [
        "python3", str(RUN_SCRIPT),
        "--workload-name", f"calib_{name}_M{M}_N{N}_K{K}",
        "--M", str(M), "--N", str(N), "--K", str(K),
        "--m-l2-spatial", str(m_l2),
        "--n-l2-spatial", str(n_l2),
        "--arch-yaml", str(ARCH_YAML),
        "--cube-k-correction", "1",     # v0.3 K=16 spatial 已建模
        "--clock-mhz", str(CLOCK_MHZ_910B4),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if res.returncode != 0:
            return None, f"Timeloop fail: {res.stderr[:200]}"
        # 解析输出找 cycles
        for line in res.stdout.splitlines():
            if 'cycles (raw):' in line:
                # "  cycles (raw):                  3,072"
                parts = line.split(':')
                if len(parts) >= 2:
                    val = parts[-1].strip().replace(',', '').replace('μs', '').strip()
                    try:
                        return int(val), None
                    except ValueError:
                        pass
        return None, "Cannot parse cycles from output"
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cube-util-json",
                   default=str(REPO / "data" / "cube_util_extracted.json"))
    p.add_argument("--output",
                   default=str(REPO / "data" / "timeloop_vs_real_calibration.json"))
    p.add_argument("--top-n", type=int, default=8,
                   help="每个 msprof 目录取 top N shapes")
    p.add_argument("--filter-msprof", default="msprof_qwen3_06b_b1_ArithmeticUtilization",
                   help="只处理某个 msprof 目录（默认 b=1 ArithUtil）")
    args = p.parse_args()

    with open(args.cube_util_json, encoding='utf-8') as f:
        data = json.load(f)

    if args.filter_msprof not in data:
        print(f"ERROR: 未找到 {args.filter_msprof}")
        print(f"可选: {list(data.keys())}")
        return 1

    # 从目录名提 batch（msprof_qwen3_06b_b4_xxx → 4）
    import re
    bm = re.search(r'_b(\d+)_', args.filter_msprof)
    batch = int(bm.group(1)) if bm else 1

    selected = data[args.filter_msprof]
    print(f"=== 校准 {args.filter_msprof} 的 top {args.top_n} shapes (batch={batch}) ===\n")

    results = []
    for s in selected['top_shapes_by_aicore_time'][:args.top_n]:
        M, N, K = s['M'], s['N'], s['K']
        if M is None or N is None or K is None:
            continue
        cube_util_real_pct = s['cube_util_pct']

        # M_effective = B × M (msprof per-op 报的是 per-batch shape，实际 invocation 内部 unroll B 个)
        M_eff = M * batch

        print(f"  shape M={M}×B={batch}={M_eff} N={N} K={K} (count={s['count']}, aicore_time={s['aicore_time_us']:.1f} μs, cube_util_real={cube_util_real_pct:.2f}%)")

        cycles_tl, err = run_timeloop_for_shape(M_eff, N, K, name=args.filter_msprof)
        if cycles_tl is None:
            print(f"    ✗ Timeloop fail: {err}")
            results.append({**s, 'timeloop_cycles': None, 'error': err})
            continue

        # 计算理论 wall_clock = cycles_tl / clock_hz
        wall_clock_theoretical_us = cycles_tl / CLOCK_MHZ_910B4   # cycles / MHz = us
        # 实测每 op 的平均 aicore_time（除算子重复次数）
        aicore_time_per_op = s['aicore_time_us'] / s['count']
        # η_real 应等于 cycles_tl / (aicore_time_per_op × clock_mhz)
        # 这等于实测 cube_util（aic_mac_fp16_ratio）
        eta_inferred = cycles_tl / (aicore_time_per_op * CLOCK_MHZ_910B4)

        print(f"    Timeloop cycles={cycles_tl}, wall_clock_theoretical={wall_clock_theoretical_us:.2f} us")
        print(f"    msprof per-op aicore_time={aicore_time_per_op:.2f} us, η_inferred={eta_inferred*100:.2f}%, η_msprof={cube_util_real_pct:.2f}%")
        if cube_util_real_pct > 0:
            print(f"    ratio (η_inferred / η_msprof) = {eta_inferred*100/cube_util_real_pct:.3f}")
        else:
            print(f"    ratio (η_inferred / η_msprof) = N/A (η_msprof = 0)")
        print()

        results.append({
            **s,
            'timeloop_cycles': cycles_tl,
            'wall_clock_theoretical_us': wall_clock_theoretical_us,
            'aicore_time_per_op_us': aicore_time_per_op,
            'eta_inferred_pct': eta_inferred * 100,
            'eta_msprof_pct': cube_util_real_pct,
            'consistency_ratio': eta_inferred / (cube_util_real_pct / 100) if cube_util_real_pct > 0 else None,
        })

    out = {
        'msprof_source': args.filter_msprof,
        'arch_yaml': str(ARCH_YAML),
        'clock_mhz': CLOCK_MHZ_910B4,
        'shapes': results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n=== 写入 {args.output}（{len(results)} 条）===")

    # 表格汇总
    print(f"\n{'M':>5} {'N':>5} {'K':>5}  {'TL cycles':>10}  {'TL wall_clock(μs)':>18}  {'msprof aicore(μs)':>18}  {'η_inferred%':>12}  {'η_msprof%':>10}  {'ratio':>6}")
    for r in results:
        if r.get('timeloop_cycles') is None:
            continue
        print(f"{r['M']:>5} {r['N']:>5} {r['K']:>5}  {r['timeloop_cycles']:>10,}  "
              f"{r['wall_clock_theoretical_us']:>18.2f}  {r['aicore_time_per_op_us']:>18.2f}  "
              f"{r['eta_inferred_pct']:>12.2f}  {r['eta_msprof_pct']:>10.2f}  {r['consistency_ratio']:>6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
