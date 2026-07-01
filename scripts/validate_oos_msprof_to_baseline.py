#!/usr/bin/env python3
"""Validate v4 prediction against a single OOS measured msprof config.

Steps:
  1. Parse msprof PipeUtilization op_summary CSV -> pipe baseline entry
  2. Read step_trace.csv to get measured wall_clock
  3. Compute kernel_gap (K0 * n_kernels) + host_gap (residual)
  4. Append to data/calibration/pipe_baseline_per_model.json
  5. Run v4 prediction with same model YAML + arch + params
  6. Print side-by-side comparison table

Generic — used for ModernBERT, Llama, SmolLM2-360M, Qwen2.5-0.5B etc.

Usage:
    python3 scripts/validate_oos_msprof_to_baseline.py \\
        --msprof-dir msprof_data/msprof_llama_3_2_1b_prefill_S2048_b1_PipeUtilization \\
        --model-yaml models/regime/llama_3_2_1b_prefill_S2048.yaml \\
        --config-name Llama-3.2-1B-prefill-S2048-b1 \\
        --loop 10
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))  # for scripts.parse_pipeutil_to_baseline

from prism.predict_pipe import ModelSpec, predict_pipe_baseline   # noqa: E402
from prism.predict_pipe.predict import _arch_dict_from_yaml        # noqa: E402
from scripts.parse_pipeutil_to_baseline import (                   # noqa: E402
    parse_pipeutil_csv, find_op_summary,
)


def find_step_trace(msprof_dir: Path) -> Path | None:
    matches = list(msprof_dir.glob("PROF_*/mindstudio_profiler_output/step_trace_*.csv"))
    return matches[0] if matches else None


def read_iter_times(step_trace: Path) -> list[float]:
    iter_times = []
    with open(step_trace, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = float(row.get("Iteration Time(us)", 0) or 0)
            if t > 0:
                iter_times.append(t)
    return iter_times


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--msprof-dir", required=True)
    p.add_argument("--model-yaml", required=True)
    p.add_argument("--config-name", required=True)
    p.add_argument("--arch-yaml", default="arch/ascend_910b4_for_sweep_v2.yaml")
    p.add_argument("--params", default="data/calibration/predict_pipe_params.json")
    p.add_argument("--baseline", default="data/calibration/pipe_baseline_per_model.json")
    p.add_argument("--loop", type=int, default=10)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    msprof_dir = Path(args.msprof_dir)

    # 1) Parse pipe breakdown
    csv_path = find_op_summary(msprof_dir)
    entry = parse_pipeutil_csv(csv_path, loop=args.loop)

    # 2) Read step_trace for measured wall_clock
    st = find_step_trace(msprof_dir)
    if st:
        iter_times = read_iter_times(st)
        mean_wall = statistics.mean(iter_times)
        std_wall = statistics.stdev(iter_times) if len(iter_times) > 1 else 0.0
        print(f"step_trace: {len(iter_times)} iterations; mean wall = {mean_wall:.1f} us (stdev {std_wall:.1f})")
    else:
        mean_wall = entry["task_dur_us"]
        print(f"WARNING: no step_trace, using task_dur_us = {mean_wall:.1f} as wall_clock proxy")

    # 3) Compute kernel_gap + host_gap using K0 from params
    params = json.load(open(args.params, encoding="utf-8"))
    K0 = params.get("K0_us_per_kernel", 1.86)
    n_kernels = entry["n_kernels_per_inf"]
    kernel_gap = K0 * n_kernels
    host_gap = mean_wall - entry["aic_time_us"] - entry["aiv_time_us"] - kernel_gap
    host_gap_per_kernel = host_gap / max(n_kernels, 1)

    entry["wall_clock_us"] = round(mean_wall, 0)
    entry["kernel_gap_us"] = round(kernel_gap, 1)
    entry["host_gap_us"] = round(host_gap, 1)
    entry["host_gap_us_per_kernel"] = round(host_gap_per_kernel, 2)

    print(f"\nMeasured entry for {args.config_name}:")
    print(json.dumps(entry, indent=2, ensure_ascii=False))

    # 4) v4 prediction
    spec = ModelSpec.from_yaml(args.model_yaml)
    arch = _arch_dict_from_yaml(args.arch_yaml)
    pred = predict_pipe_baseline(spec, arch, params, batch=args.batch)

    # 5) Compare
    print("\n" + "=" * 70)
    print(f"v4 Prediction vs Measured: {args.config_name}")
    print("=" * 70)
    print(f"{'Field':<22s} {'Pred (v4)':>14s} {'Measured':>14s} {'Err %':>10s}")
    print("-" * 70)
    fields = [
        "n_kernels_per_inf", "aic_time_us", "aiv_time_us",
        "kernel_gap_us", "host_gap_us", "wall_clock_us",
    ]
    for k in fields:
        p_val = pred[k]
        m_val = entry[k]
        err = (p_val - m_val) / m_val * 100 if m_val else 0.0
        print(f"{k:<22s} {p_val:>14.0f} {m_val:>14.0f} {err:>+9.1f}%")
    print()
    print(f"aic_dominant: pred={pred['aic_dominant_pipe']:s}  measured={entry['aic_dominant_pipe']:s}")
    print(f"aic_pipes_us pred: {dict((k, round(v,0)) for k,v in pred['aic_pipes_us'].items())}")
    print(f"aic_pipes_us meas: {dict((k, round(v,0)) for k,v in entry['aic_pipes_us'].items())}")
    print(f"aiv_pipes_us meas: {dict((k, round(v,0)) for k,v in entry['aiv_pipes_us'].items())}")
    print(f"confidence (v4 says): {pred['confidence']}")

    # 6) Merge into baseline JSON
    if not args.dry_run:
        with open(args.baseline, encoding="utf-8") as f:
            doc = json.load(f)
        doc["configs"][args.config_name] = entry
        with open(args.baseline, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"\nMerged into {args.baseline} ({len(doc['configs'])} configs total)")


if __name__ == "__main__":
    main()
