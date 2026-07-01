#!/usr/bin/env python3
"""Parse msprof PipeUtilization op_summary CSV → pipe_baseline_per_model.json entry.

Aggregates per-op (per task) pipe times into per-inference totals, matching
the schema used by ``data/calibration/pipe_baseline_per_model.json``.

Usage:
    python3 scripts/parse_pipeutil_to_baseline.py \\
        --msprof-dir msprof_data/msprof_bert_base_b4_PipeUtilization \\
        --loop 10 \\
        --config-name BERT-base-S128-b4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple


# Per-kernel CANN runtime cost; calibrated value from
# data/calibration/predict_pipe_params.json::K0_us_per_kernel (see methodology/08 §4.1).
K0_US_PER_KERNEL = 1.8557901133663446


def _to_float(x: str) -> float:
    try:
        return float(x or 0)
    except (ValueError, TypeError):
        return 0.0


def parse_pipeutil_csv(csv_path: Path, loop: int = 10) -> Dict:
    """Aggregate one PipeUtilization op_summary CSV into pipe_baseline entry shape."""
    aic_pipes = {"mac": 0.0, "mte1": 0.0, "mte2": 0.0, "fixpipe": 0.0, "scalar": 0.0}
    aiv_pipes = {"vec": 0.0, "mte2": 0.0, "mte3": 0.0, "scalar": 0.0}
    aic_time_total = 0.0
    aiv_time_total = 0.0
    task_dur_total = 0.0
    n_rows = 0

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            op_type = (row.get("OP Type") or "").strip()
            # Skip profiling-meta rows
            if op_type in ("", "PROFILING_ENABLE", "MODEL_MAINTAINCE"):
                continue
            aic = _to_float(row.get("aicore_time(us)"))
            aiv = _to_float(row.get("aiv_time(us)"))
            if aic == 0 and aiv == 0:
                continue

            aic_pipes["mac"]     += _to_float(row.get("aic_mac_time(us)"))
            aic_pipes["mte1"]    += _to_float(row.get("aic_mte1_time(us)"))
            aic_pipes["mte2"]    += _to_float(row.get("aic_mte2_time(us)"))
            aic_pipes["fixpipe"] += _to_float(row.get("aic_fixpipe_time(us)"))
            aic_pipes["scalar"]  += _to_float(row.get("aic_scalar_time(us)"))

            aiv_pipes["vec"]     += _to_float(row.get("aiv_vec_time(us)"))
            aiv_pipes["mte2"]    += _to_float(row.get("aiv_mte2_time(us)"))
            aiv_pipes["mte3"]    += _to_float(row.get("aiv_mte3_time(us)"))
            aiv_pipes["scalar"]  += _to_float(row.get("aiv_scalar_time(us)"))

            aic_time_total += aic
            aiv_time_total += aiv
            task_dur_total += _to_float(row.get("Task Duration(us)"))
            n_rows += 1

    # Divide by loop count to get per-inference values
    L = max(int(loop), 1)
    for k in aic_pipes: aic_pipes[k] /= L
    for k in aiv_pipes: aiv_pipes[k] /= L
    aic_time = aic_time_total / L
    aiv_time = aiv_time_total / L
    task_dur = task_dur_total / L
    n_kernels = n_rows // L

    # AIV idle = aiv_time - (vec + mte2 + mte3 + scalar)
    aiv_idle = max(0.0, aiv_time - sum(aiv_pipes.values()))
    aiv_pipes_out = dict(aiv_pipes)
    aiv_pipes_out["idle"] = aiv_idle

    # AIC bubble = aic_time - max(aic_pipes) (similar to aiv_idle but on dominant pipe)
    aic_dominant_pipe = max(aic_pipes, key=lambda k: aic_pipes[k])
    aic_bubble = max(0.0, aic_time - max(aic_pipes.values()))

    return {
        "n_kernels_per_inf": int(n_kernels),
        "task_dur_us": round(task_dur, 1),
        "aic_time_us": round(aic_time, 1),
        "aiv_time_us": round(aiv_time, 1),
        "aic_pipes_us": {k: round(v, 1) for k, v in aic_pipes.items()},
        "aiv_pipes_us": {k: round(v, 1) for k, v in aiv_pipes_out.items()},
        "aic_bubble_us": round(aic_bubble, 1),
        "aic_dominant_pipe": aic_dominant_pipe,
        "source": "msprof_PipeUtilization_measured",
    }


def find_op_summary(msprof_dir: Path) -> Path:
    """Find op_summary*.csv under msprof_dir/PROF_*/mindstudio_profiler_output/."""
    matches = list(msprof_dir.glob("PROF_*/mindstudio_profiler_output/op_summary*.csv"))
    if not matches:
        raise FileNotFoundError(f"No op_summary*.csv under {msprof_dir}/PROF_*/")
    return matches[0]


def find_step_trace(msprof_dir: Path) -> "Path|None":
    """Find step_trace*.csv (the per-iteration wall-clock source). May be absent for
    decode workloads — msprof sometimes omits it (see methodology/05 §6.4)."""
    matches = list(msprof_dir.glob("PROF_*/mindstudio_profiler_output/step_trace*.csv"))
    return matches[0] if matches else None


def read_wall_clock_us(step_trace_path: Path) -> float:
    """Mean per-iteration wall-clock (μs) from a step_trace CSV."""
    times = []
    with open(step_trace_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("Iteration Time(us)") or "").strip()
            if v:
                times.append(_to_float(v))
    return sum(times) / len(times) if times else 0.0


def fill_wall_clock_gaps(entry: Dict, wall_clock_us: float,
                         k0_us: float = K0_US_PER_KERNEL) -> Dict:
    """Split the wall-clock overhead into kernel_gap + host_gap so the entry is a
    complete pipe_baseline_per_model.json record consumable by prism-ceiling / prism-sweep.

    overhead = wall_clock - aic_time - aiv_time   (serial model, see methodology/02)
      kernel_gap = min(overhead, k0 × n_kernels)
      host_gap   = overhead - kernel_gap
    """
    out = dict(entry)
    n_kernels = max(int(entry.get("n_kernels_per_inf", 0)), 0)
    overhead = max(0.0, wall_clock_us - entry["aic_time_us"] - entry["aiv_time_us"])
    kernel_gap = min(overhead, k0_us * n_kernels)
    host_gap = overhead - kernel_gap
    out["wall_clock_us"] = round(wall_clock_us, 1)
    out["kernel_gap_us"] = round(kernel_gap, 1)
    out["host_gap_us"] = round(host_gap, 1)
    out["host_gap_us_per_kernel"] = round(host_gap / n_kernels, 2) if n_kernels else 0.0
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--msprof-dir", required=True,
                   help="Directory like msprof_bert_base_b4_PipeUtilization/")
    p.add_argument("--loop", type=int, default=10, help="ais_bench loop count (default: 10)")
    p.add_argument("--config-name", required=True,
                   help='Output JSON key, e.g. "BERT-base-S128-b4"')
    p.add_argument("--merge-into", default="data/calibration/pipe_baseline_per_model.json",
                   help="Pipe baseline JSON file to merge into (in-place).")
    p.add_argument("--wall-clock-us", type=float, default=None,
                   help="Per-inference wall-clock (μs). Overrides step_trace; use when "
                        "step_trace*.csv is absent (e.g. decode workloads).")
    p.add_argument("--k0", type=float, default=K0_US_PER_KERNEL,
                   help=f"Per-kernel CANN runtime cost μs (default: {K0_US_PER_KERNEL:.4f}).")
    p.add_argument("--dry-run", action="store_true", help="Print, don't write.")
    args = p.parse_args()

    msprof_dir = Path(args.msprof_dir)
    csv_path = find_op_summary(msprof_dir)
    entry = parse_pipeutil_csv(csv_path, loop=args.loop)

    # Fill wall_clock_us / kernel_gap_us / host_gap_us so the entry is a complete
    # pipe_baseline record (prism-ceiling / prism-sweep need these). Precedence:
    # --wall-clock-us > step_trace CSV > 0.0 placeholders.
    wall_clock_us = args.wall_clock_us
    if wall_clock_us is None:
        step_trace = find_step_trace(msprof_dir)
        if step_trace is not None:
            wall_clock_us = read_wall_clock_us(step_trace)

    if wall_clock_us and wall_clock_us > 0:
        entry_with_kg_hg = fill_wall_clock_gaps(entry, wall_clock_us, k0_us=args.k0)
    else:
        print("  ⚠ step_trace*.csv 缺失且未给 --wall-clock-us；"
              "wall_clock/kernel_gap/host_gap 留 0 占位（prism-ceiling 的 S0 将不可信）")
        entry_with_kg_hg = dict(entry)
        entry_with_kg_hg["wall_clock_us"] = 0
        entry_with_kg_hg["kernel_gap_us"] = 0.0
        entry_with_kg_hg["host_gap_us"] = 0.0
        entry_with_kg_hg["host_gap_us_per_kernel"] = 0.0

    print(f"=== {args.config_name} from {csv_path.name} (loop={args.loop}) ===")
    print(json.dumps(entry_with_kg_hg, indent=2, ensure_ascii=False))

    if args.dry_run:
        return 0

    out_path = Path(args.merge_into)
    with open(out_path, encoding="utf-8") as f:
        doc = json.load(f)
    if "configs" not in doc:
        doc = {"configs": doc}
    doc["configs"][args.config_name] = entry_with_kg_hg
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"\nMerged → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
