#!/usr/bin/env python3
"""
parse_timeloop_stats.py — Timeloop stats.txt 解析器
Phase F1/F4/F5：解析 timeloop-mapper 输出，提取延迟/能效/面积数据

用法：
  python3 benchmark/parse_timeloop_stats.py \
      --stats-dir timeloop_results/bert_base_910b4/ \
      --model bert_base --chip 910b4 \
      --msprof-npu-ms 1.826 \
      --out-json timeloop_results/bert_base_910b4_summary.json

输出字段说明：
  cycles_total         : 所有 GEMM 算子的 cycles 加总
  energy_total_pj      : 所有 GEMM 算子的能耗加总（pJ）
  area_mm2             : 芯片面积估算（mm²，取各 op 最大值）
  latency_gemm_us      : GEMM 路径延迟（μs），= cycles × clock_period
  latency_npu_msprof_us: msprof 实测 NPU op time（μs）
  bias_vs_msprof_pct   : Timeloop vs msprof 偏差（%）
  ops_breakdown        : 每个 GEMM 算子的详细 cycles/energy/utilization
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ── 芯片时钟频率（Hz） ──────────────────────────────────────────────
CHIP_CLOCK_HZ = {
    "910b4": 1_500_000_000,   # 1500 MHz（Cube Core，910B4）
    "310p":  1_000_000_000,   # 1000 MHz（310P DaVinci Lite）
}

# ── 算子列表（bert_base / hf_bert 定义的顺序） ──────────────────────
BERT_BASE_OPS = [
    "q_projection", "k_projection", "v_projection",
    "attention_qkt", "attention_av", "output_projection",
    "ffn_linear1", "ffn_linear2",
]
HF_BERT_OPS = [
    "q_projection", "k_projection", "v_projection",
    "attention_qkt", "attention_av", "output_projection",
    "ffn_linear1", "ffn_linear2", "pooler",
]


def parse_stats_file(stats_path: Path) -> dict:
    """
    解析单个 timeloop-mapper.stats.txt 文件。

    Timeloop stats.txt 典型格式（Summary Stats 区域）：
      GFLOPs = 25.17
      Utilization = 0.1875
      Cycles = 27295744
      Energy = 1234567.89 pJ
      EDP = ...
      Area = 12.34 mm^2
    """
    if not stats_path.exists():
        return {"error": f"File not found: {stats_path}"}

    text = stats_path.read_text(encoding="utf-8", errors="replace")
    result = {
        "source_file": str(stats_path),
        "cycles":      None,
        "energy_pj":   None,
        "area_mm2":    None,
        "utilization": None,
        "gflops":      None,
        "edp":         None,
    }

    patterns = {
        "cycles":      r"Cycles\s*=\s*([0-9]+)",
        "energy_pj":   r"Energy\s*=\s*([0-9.eE+\-]+)\s*pJ",
        "area_mm2":    r"Area\s*=\s*([0-9.eE+\-]+)\s*mm\^2",
        "utilization": r"Utilization\s*=\s*([0-9.eE+\-]+)",
        "gflops":      r"GFLOPs?\s*=\s*([0-9.eE+\-]+)",
        "edp":         r"EDP\s*=\s*([0-9.eE+\-]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            result[key] = float(m.group(1))

    # 备用格式：Total cycles / Total Energy
    if result["cycles"] is None:
        m = re.search(r"[Tt]otal\s+[Cc]ycles[:\s]+([0-9]+)", text)
        if m:
            result["cycles"] = float(m.group(1))
    if result["energy_pj"] is None:
        m = re.search(r"[Tt]otal\s+[Ee]nergy[:\s]+([0-9.eE+\-]+)\s*pJ", text)
        if m:
            result["energy_pj"] = float(m.group(1))

    return result


def find_stats_files(stats_dir: Path, op_names: list) -> dict:
    """
    在 stats_dir 下查找每个算子的 stats 文件。

    支持目录结构：
      1. stats_dir/<op_name>/timeloop-mapper.stats.txt
      2. stats_dir/<op_name>/timeloop-model.stats.txt
      3. stats_dir/<op_name>.stats.txt
      4. stats_dir/timeloop-mapper.stats.txt  (单算子回退)
    """
    found = {}
    for op in op_names:
        c1 = stats_dir / op / "timeloop-mapper.stats.txt"
        c2 = stats_dir / op / "timeloop-model.stats.txt"
        c3 = stats_dir / f"{op}.stats.txt"
        c4 = stats_dir / "timeloop-mapper.stats.txt"

        for c in [c1, c2, c3]:
            if c.exists():
                found[op] = c
                break
        if op not in found and c4.exists() and len(op_names) == 1:
            found[op] = c4
    return found


def compute_latency_us(cycles: float, chip: str) -> float:
    clock_hz = CHIP_CLOCK_HZ.get(chip, 1_000_000_000)
    return cycles / clock_hz * 1e6


def summarize_ops(op_stats: dict, chip: str, num_layers: int = 1) -> dict:
    total_cycles    = 0
    total_energy_pj = 0.0
    max_area_mm2    = 0.0
    valid_ops       = 0
    breakdown       = {}

    for op_name, stats in op_stats.items():
        if stats.get("cycles") is None:
            breakdown[op_name] = {"error": "cycles not found", **stats}
            continue
        cycles     = stats["cycles"]
        latency_us = compute_latency_us(cycles, chip)
        energy_pj  = stats.get("energy_pj", 0.0) or 0.0
        area_mm2   = stats.get("area_mm2",  0.0) or 0.0

        total_cycles    += cycles
        total_energy_pj += energy_pj
        if area_mm2 > max_area_mm2:
            max_area_mm2 = area_mm2
        valid_ops += 1

        breakdown[op_name] = {
            "cycles":      int(cycles),
            "latency_us":  round(latency_us, 2),
            "energy_pj":   round(energy_pj, 2),
            "area_mm2":    round(area_mm2, 4),
            "utilization": stats.get("utilization"),
            "gflops":      stats.get("gflops"),
        }

    total_latency_us          = compute_latency_us(total_cycles, chip)
    total_latency_per_layer_us = (total_latency_us / num_layers
                                  if num_layers > 1 else total_latency_us)

    return {
        "chip":                  chip,
        "num_layers":            num_layers,
        "valid_ops":             valid_ops,
        "total_ops":             len(op_stats),
        "cycles_total":          int(total_cycles),
        "latency_gemm_us":       round(total_latency_us, 2),
        "latency_per_layer_us":  round(total_latency_per_layer_us, 2),
        "energy_total_pj":       round(total_energy_pj, 2),
        "area_mm2":              round(max_area_mm2, 4),
        "ops_breakdown":         breakdown,
    }


def compare_with_msprof(summary: dict, msprof_npu_ms: float) -> dict:
    """将 Timeloop GEMM 延迟与 msprof 实测 NPU op time 对比。"""
    tl_us = summary["latency_gemm_us"]
    ms_us = msprof_npu_ms * 1000.0

    # Timeloop 只建模 Cube GEMM，不含 Vector（Softmax/LayerNorm/GELU）
    # 估算 Vector 占比约 15%（基于 msprof PipeUtilization 数据）
    vector_fraction       = 0.15
    estimated_cube_us     = ms_us * (1 - vector_fraction)

    bias_vs_total_pct = (tl_us - ms_us)          / ms_us          * 100
    bias_vs_cube_pct  = (tl_us - estimated_cube_us) / estimated_cube_us * 100

    return {
        "timeloop_gemm_us":         round(tl_us, 2),
        "msprof_npu_total_us":      round(ms_us, 2),
        "msprof_cube_estimated_us": round(estimated_cube_us, 2),
        "bias_vs_total_pct":        round(bias_vs_total_pct, 1),
        "bias_vs_cube_pct":         round(bias_vs_cube_pct, 1),
        "pass_25pct_threshold":     abs(bias_vs_cube_pct) <= 25.0,
        "note": (
            "Timeloop models only Cube GEMM; vector ops (Softmax/LayerNorm/GELU) "
            f"estimated at {vector_fraction*100:.0f}% of total NPU time. "
            "bias_vs_cube_pct is the physically meaningful comparison."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="解析 timeloop-mapper stats.txt，生成汇总 JSON"
    )
    parser.add_argument("--stats-dir", required=True,
                        help="含 Timeloop stats 文件的目录（每算子一个子目录）")
    parser.add_argument("--model", default="bert_base",
                        choices=["bert_base", "hf_bert", "gpt2_small",
                                 "qwen3_06b", "net_transformer"],
                        help="模型名称（决定算子列表）")
    parser.add_argument("--chip", default="910b4",
                        choices=list(CHIP_CLOCK_HZ.keys()),
                        help="芯片型号（决定时钟频率）")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="模型层数（用于 per-layer 延迟计算，默认 1）")
    parser.add_argument("--msprof-npu-ms", type=float, default=None,
                        help="msprof 实测 NPU op time（ms），用于偏差对比")
    parser.add_argument("--out-json", default=None,
                        help="输出 JSON 路径（默认打印到 stdout）")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    stats_dir = Path(args.stats_dir)
    if not stats_dir.exists():
        print(f"ERROR: stats-dir not found: {stats_dir}", file=sys.stderr)
        sys.exit(1)

    op_map = {
        "bert_base":       BERT_BASE_OPS,
        "hf_bert":         HF_BERT_OPS,
        "gpt2_small":      BERT_BASE_OPS,
        "qwen3_06b":       BERT_BASE_OPS,
        "net_transformer": BERT_BASE_OPS,
    }
    op_names  = op_map.get(args.model, BERT_BASE_OPS)
    stats_files = find_stats_files(stats_dir, op_names)

    if not stats_files:
        print(f"ERROR: No stats files found in {stats_dir}", file=sys.stderr)
        sys.exit(1)

    op_stats = {}
    for op_name in op_names:
        if op_name in stats_files:
            op_stats[op_name] = parse_stats_file(stats_files[op_name])
            if args.verbose:
                cyc = op_stats[op_name].get("cycles")
                eng = op_stats[op_name].get("energy_pj") or 0.0
                print(f"  {op_name}: cycles={cyc}, energy={eng:.1f} pJ")
        else:
            op_stats[op_name] = {"error": "stats file not found"}

    summary              = summarize_ops(op_stats, args.chip, args.num_layers)
    summary["model"]     = args.model
    summary["stats_dir"] = str(stats_dir)
    summary["clock_hz"]  = CHIP_CLOCK_HZ[args.chip]

    if args.msprof_npu_ms is not None:
        summary["msprof_comparison"] = compare_with_msprof(summary, args.msprof_npu_ms)

    out_json = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json)
        print(f"✓ 写入 {out_path}")
    else:
        print(out_json)

    # ── 控制台摘要 ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Timeloop 解析摘要  model={args.model}  chip={args.chip}")
    print("="*60)
    print(f"  有效算子数：{summary['valid_ops']} / {summary['total_ops']}")
    print(f"  总 cycles：{summary['cycles_total']:,}")
    print(f"  GEMM 路径延迟：{summary['latency_gemm_us']:.1f} μs")
    if args.num_layers > 1:
        print(f"  Per-layer 延迟：{summary['latency_per_layer_us']:.1f} μs/层")
    print(f"  总能耗：{summary['energy_total_pj']:.1f} pJ")
    print(f"  芯片面积估算：{summary['area_mm2']:.2f} mm²")

    if "msprof_comparison" in summary:
        mc     = summary["msprof_comparison"]
        status = "✅ PASS" if mc["pass_25pct_threshold"] else "⚠ WARN"
        print(f"\n  msprof 对比：")
        print(f"    Timeloop GEMM：    {mc['timeloop_gemm_us']:.1f} μs")
        print(f"    msprof NPU total： {mc['msprof_npu_total_us']:.1f} μs")
        print(f"    msprof Cube 估算： {mc['msprof_cube_estimated_us']:.1f} μs")
        print(f"    偏差（vs total）：{mc['bias_vs_total_pct']:+.1f}%")
        print(f"    偏差（vs cube）：  {mc['bias_vs_cube_pct']:+.1f}%  {status}（目标 ±25%）")


if __name__ == "__main__":
    main()
