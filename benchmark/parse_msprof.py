#!/usr/bin/env python3
"""
parse_msprof.py — 解析 msprof 产物 CSV，输出 per-model 物理分解 JSON

支持的 aic-metrics group：PipeUtilization / L2Cache / Memory / ArithmeticUtilization

用法：
  python3 parse_msprof.py --all     # 解析 msprof_data/ 下全部模型
  python3 parse_msprof.py --model bert_base --batch 1 --metric PipeUtilization
  python3 parse_msprof.py --model bert_base --batch 1 --metric all  # 合并 4 个 metric

  # Cube/Vector/MTE 算子分解（独立工具，直接消费 op_summary.csv）
  python3 parse_msprof.py --vector-mte-csv path/to/op_summary.csv \
                          --vector-mte-json out/vector_mte.json

输出：
  msprof_data/<model>_b<batch>.json  — 合并后的物理分解
  docs/msprof_breakdown_<model>.md   — markdown 报告

新增（2026-05-06，Phase G+ Track B）：
  extract_vector_mte_decomposition(op_summary_csv_path) 函数
    输入：单个 msprof op_summary.csv 路径
    输出：dict 含 total_{cube|vector|mte|other}_us、对应 fraction，以及 by_op_type
          细分（每个 op_type 的 count 与 total_us）。
    分类启发式：基于 OP Type / Op Name 关键字（CUBE_OPS / VECTOR_OPS /
                 MTE_OPS / EMBEDDING_OPS 集合）。
    用途：为 Cube 路径修正系数（见 docs/empirical_cube_correction.md）提供
          Vector/MTE 旁路时间，使得
              T_total = T_cube_corrected + T_vector + T_mte + β_overhead
          可分项重建。

  CLI flag --vector-mte-json <output_path>：将 extract_vector_mte_decomposition
  返回 dict 写入指定路径，供下游脚本（roofline 校准、修正系数应用等）消费。
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"


def resolve_msprof_dir(cli_override: "str|None" = None) -> Path:
    """Locate the msprof_data root. Precedence: CLI flag > MSPROF_BASE env > repo default.

    Lets downstream projects point PRISM at their own msprof_data without symlink hacks.
    """
    if cli_override:
        return Path(cli_override).expanduser().resolve()
    env = os.environ.get("MSPROF_BASE")
    if env:
        return Path(env).expanduser().resolve()
    return ROOT / "msprof_data"


MSPROF_DIR = resolve_msprof_dir()

# 算子分类
CUBE_OPS = {"MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2", "Conv2D", "Conv3D", "Gemm"}
VECTOR_OPS = {"Softmax", "SoftmaxV2", "LayerNorm", "LayerNormV2", "Add", "Mul", "Sub", "Div",
              "Gelu", "Relu", "Tanh", "Sigmoid", "Cast", "Sqrt", "Rsqrt", "Pow", "Erf",
              "ReduceSum", "ReduceMean", "ReduceMax", "Square", "RealDiv", "FastGelu"}
MTE_OPS = {"TransData", "Transpose", "Reshape", "Slice", "Concat", "ConcatD", "Split", "Stack",
           "BroadcastTo", "Tile", "Expand", "Where"}
EMBEDDING_OPS = {"GatherV2", "Gather", "OneHot", "Embedding"}

LAYER_REGEX = [
    re.compile(r"layer[_./](\d+)", re.IGNORECASE),
    re.compile(r"transformer\.h\.(\d+)\.", re.IGNORECASE),
    re.compile(r"\.h\.(\d+)\.", re.IGNORECASE),
    re.compile(r"_h_(\d+)_", re.IGNORECASE),
    re.compile(r"layers?\.(\d+)", re.IGNORECASE),
]


def find_prof_dir(model: str, batch: int, metric: str) -> "Path|None":
    """定位某个 metric 的 mindstudio_profiler_output 目录"""
    pattern = f"msprof_{model}_b{batch}_{metric}/PROF_*/mindstudio_profiler_output"
    matches = list(MSPROF_DIR.glob(pattern))
    return matches[0] if matches else None


def load_csv(prof_dir: Path, prefix: str) -> list[dict]:
    matches = list(prof_dir.glob(f"{prefix}_*.csv"))
    if not matches:
        return []
    with open(matches[0], encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify_op(op_type: str, op_name: str) -> str:
    if op_type in CUBE_OPS:
        return "cube"
    if op_type in VECTOR_OPS:
        return "vector"
    if op_type in MTE_OPS:
        return "mte"
    if op_type in EMBEDDING_OPS:
        return "embedding"
    return "other"


def extract_layer(op_name: str):
    for r in LAYER_REGEX:
        m = r.search(op_name)
        if m:
            return int(m.group(1))
    return None


def is_attention_op(op_name: str, op_type: str) -> bool:
    """识别 attention 路径上的算子"""
    name_lower = op_name.lower()
    if op_type in {"BatchMatMul", "BatchMatMulV2"}:
        return True
    if op_type in {"Softmax", "SoftmaxV2"} and "attn" in name_lower or "attention" in name_lower:
        return True
    return ("attn" in name_lower or "attention" in name_lower) and op_type in CUBE_OPS | VECTOR_OPS


def is_ffn_op(op_name: str, op_type: str) -> bool:
    name_lower = op_name.lower()
    return ("ffn" in name_lower or "mlp" in name_lower or "feed" in name_lower or "intermediate" in name_lower) \
           and op_type in CUBE_OPS


def parse_model_metric(model: str, batch: int, metric: str):
    prof_dir = find_prof_dir(model, batch, metric)
    if not prof_dir:
        return None

    # step_trace: 每 iteration 的 wall-clock 时间
    step = load_csv(prof_dir, "step_trace")
    iter_times_us = [float(r["Iteration Time(us)"]) for r in step if r["Iteration Time(us)"].strip()]
    n_iter = len(iter_times_us)
    iter_avg_us = statistics.mean(iter_times_us) if iter_times_us else 0.0
    iter_std_us = statistics.stdev(iter_times_us) if len(iter_times_us) > 1 else 0.0

    # op_summary
    ops = load_csv(prof_dir, "op_summary")
    op_total_us = 0.0
    op_by_class = defaultdict(float)
    op_by_type = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    op_by_layer = defaultdict(float)
    cube_util_per_op = []
    mac_ratio_weighted = 0.0
    vec_ratio_weighted = 0.0
    mte1_ratio_weighted = 0.0
    mte2_ratio_weighted = 0.0
    mte3_ratio_weighted = 0.0
    aicore_time_total = 0.0
    aiv_time_total = 0.0
    aic_mac_time_total = 0.0
    aic_mte2_time_total = 0.0  # MTE2 = GM↔L1 数据搬运
    attn_ops_us = 0.0
    ffn_ops_us = 0.0

    for r in ops:
        try:
            dur = float(r["Task Duration(us)"])
        except Exception:
            continue
        if dur <= 0:
            continue
        op_total_us += dur
        op_type = r.get("OP Type", "")
        op_name = r.get("Op Name", "")
        cls = classify_op(op_type, op_name)
        op_by_class[cls] += dur
        op_by_type[op_type]["count"] += 1
        op_by_type[op_type]["total_us"] += dur

        layer = extract_layer(op_name)
        if layer is not None:
            op_by_layer[layer] += dur
        else:
            op_by_layer["prelude_postlude"] += dur

        # AI core metrics（PipeUtilization 数据）
        try:
            aic_t = float(r.get("aicore_time(us)", 0))
            aiv_t = float(r.get("aiv_time(us)", 0))
            aicore_time_total += aic_t
            aiv_time_total += aiv_t
            mac_t = float(r.get("aic_mac_time(us)", 0))
            aic_mac_time_total += mac_t
            mte2_t = float(r.get("aic_mte2_time(us)", 0))
            aic_mte2_time_total += mte2_t
            cube_u = float(r.get("cube_utilization(%)", 0))
            cube_util_per_op.append((dur, cube_u))
        except Exception:
            pass

        if is_attention_op(op_name, op_type):
            attn_ops_us += dur
        if is_ffn_op(op_name, op_type):
            ffn_ops_us += dur

    op_per_iter_us = op_total_us / max(n_iter, 1)
    host_overhead_us = iter_avg_us - op_per_iter_us

    # api_statistic: host 端 API 时间
    apis = load_csv(prof_dir, "api_statistic")
    api_breakdown = {}
    api_total_us_per_iter = {}
    for r in apis:
        api_name = r.get("API Name", "")
        try:
            t_us = float(r.get("Time(us)", 0))
            n = int(r.get("Count", 0))
        except Exception:
            continue
        api_breakdown[api_name] = {"total_us": t_us, "count": n}

    # 关键 host API（每 iteration 平均）
    def get_per_iter(name):
        d = api_breakdown.get(name, {"total_us": 0, "count": 0})
        return d["total_us"] / max(n_iter, 1)

    h2d_per_iter = get_per_iter("MemCopySync") + get_per_iter("aclrtMemcpy")  # H2D + D2H
    sync_per_iter = get_per_iter("StreamSyncTaskFinish") + get_per_iter("aclrtSynchronizeStream")
    exec_per_iter = get_per_iter("aclmdlExecute")
    runtime_exec_per_iter = get_per_iter("ModelExecute")

    # L2 cache（来自 l2_cache CSV，仅 L2Cache metric 时有命中率字段）
    l2 = load_csv(prof_dir, "l2_cache")
    l2_hit_rates = []
    l2_victim_rates = []
    for r in l2:
        hr = r.get("Hit Rate", "")
        vr = r.get("Victim Rate", "")
        try:
            if hr and hr.upper() not in ["N/A", "NAN", ""]:
                l2_hit_rates.append(float(hr))
            if vr and vr.upper() not in ["N/A", "NAN", ""]:
                l2_victim_rates.append(float(vr))
        except Exception:
            pass

    # Memory metric 字段（main_mem_read/write_bw 在 op_summary 里）
    main_mem_read_bw_avg = 0.0
    main_mem_write_bw_avg = 0.0
    if metric == "Memory":
        # Memory metric 的 op_summary 包含 main_mem_*_bw 字段
        bw_vals = []
        for r in ops:
            try:
                rb = float(r.get("ai*_main_mem_read_bw(GB/s)", 0)) or float(r.get("aic_main_mem_read_bw(GB/s)", 0))
                wb = float(r.get("ai*_main_mem_write_bw(GB/s)", 0)) or float(r.get("aic_main_mem_write_bw(GB/s)", 0))
                if rb > 0 or wb > 0:
                    bw_vals.append((rb, wb))
            except Exception:
                pass
        if bw_vals:
            main_mem_read_bw_avg = statistics.mean(r for r, _ in bw_vals)
            main_mem_write_bw_avg = statistics.mean(w for _, w in bw_vals)

    return {
        "model": model,
        "batch": batch,
        "metric": metric,
        "n_iter": n_iter,
        "wall_clock": {
            "iter_avg_us": round(iter_avg_us, 2),
            "iter_std_us": round(iter_std_us, 2),
            "iter_avg_ms": round(iter_avg_us / 1000, 4),
        },
        "npu_op_sum": {
            "per_iter_us": round(op_per_iter_us, 2),
            "per_iter_ms": round(op_per_iter_us / 1000, 4),
            "by_class_us": {k: round(v / n_iter, 2) for k, v in op_by_class.items()},
            "by_layer_us": {str(k): round(v / n_iter, 2) for k, v in sorted(op_by_layer.items(), key=lambda x: (str(x[0])))},
            "attention_path_us": round(attn_ops_us / n_iter, 2),
            "ffn_path_us": round(ffn_ops_us / n_iter, 2),
        },
        "host_overhead": {
            "total_us": round(host_overhead_us, 2),
            "total_ms": round(host_overhead_us / 1000, 4),
            "fraction_of_wall": round(host_overhead_us / iter_avg_us, 3) if iter_avg_us > 0 else 0,
            "h2d_d2h_per_iter_us": round(h2d_per_iter, 2),
            "sync_per_iter_us": round(sync_per_iter, 2),
            "aclmdlExecute_per_iter_us": round(exec_per_iter, 2),
            "ModelExecute_runtime_per_iter_us": round(runtime_exec_per_iter, 2),
        },
        "compute_metrics": {
            "aicore_time_per_iter_us": round(aicore_time_total / n_iter, 2),
            "aiv_time_per_iter_us": round(aiv_time_total / n_iter, 2),
            "aic_mac_time_per_iter_us": round(aic_mac_time_total / n_iter, 2),
            "aic_mte2_time_per_iter_us": round(aic_mte2_time_total / n_iter, 2),
            "cube_util_avg_pct": round(statistics.mean([u for _, u in cube_util_per_op]), 3) if cube_util_per_op else 0,
            "cube_util_weighted_pct": round(sum(d * u for d, u in cube_util_per_op) / sum(d for d, _ in cube_util_per_op), 3) if cube_util_per_op else 0,
            "mac_compute_fraction": round(aic_mac_time_total / aicore_time_total, 3) if aicore_time_total > 0 else 0,
            "mte2_fraction_of_aic": round(aic_mte2_time_total / aicore_time_total, 3) if aicore_time_total > 0 else 0,
        },
        "l2_cache": {
            "rows": len(l2),
            "rows_with_hit_rate": len(l2_hit_rates),
            "avg_hit_rate": round(statistics.mean(l2_hit_rates), 3) if l2_hit_rates else None,
            "avg_victim_rate": round(statistics.mean(l2_victim_rates), 3) if l2_victim_rates else None,
        },
        "hbm_bandwidth": {
            "main_mem_read_bw_GBs": round(main_mem_read_bw_avg, 2),
            "main_mem_write_bw_GBs": round(main_mem_write_bw_avg, 2),
            "total_bw_GBs": round(main_mem_read_bw_avg + main_mem_write_bw_avg, 2),
            "utilization_vs_392": round((main_mem_read_bw_avg + main_mem_write_bw_avg) / 392, 3),
        },
        "top_op_types": [
            {"type": t, "count": d["count"], "total_per_iter_us": round(d["total_us"] / n_iter, 2)}
            for t, d in sorted(op_by_type.items(), key=lambda x: -x[1]["total_us"])[:10]
        ],
        "host_api_top10": [
            {"name": k, "total_us": v["total_us"], "count": v["count"], "avg_us": v["total_us"] / max(v["count"], 1)}
            for k, v in sorted(api_breakdown.items(), key=lambda x: -x[1]["total_us"])[:10]
        ],
    }


def extract_vector_mte_decomposition(op_summary_csv_path):
    """
    解析单个 msprof op_summary.csv，按 Cube / Vector / MTE / Other 分类聚合
    Task Duration(us)，并提供每个 op_type 的细分。

    Args:
        op_summary_csv_path: 字符串或 Path，指向 op_summary_*.csv 文件

    Returns:
        dict，结构：
            {
              "source_csv":         <str>,
              "rows_total":         <int>,           # CSV 行数（含负/零时长）
              "rows_counted":       <int>,           # 实际计入聚合的行数
              "total_cube_us":      <float>,
              "total_vector_us":    <float>,
              "total_mte_us":       <float>,
              "total_other_us":     <float>,
              "total_us":           <float>,         # 上述 4 项之和
              "cube_fraction":      <float>,         # 0–1.0
              "vector_fraction":    <float>,
              "mte_fraction":       <float>,
              "other_fraction":     <float>,
              "by_op_type": {
                "<OP Type>": {
                    "category":  "cube|vector|mte|embedding|other",
                    "count":     <int>,
                    "total_us":  <float>,
                    "avg_us":    <float>,
                },
                ...
              }
            }

    分类规则：
        - Cube  : OP Type ∈ CUBE_OPS（MatMul / BatchMatMul / Conv2D 等）
        - Vector: OP Type ∈ VECTOR_OPS（LayerNorm / Softmax / GeLU / Cast / Add / Mul 等）
        - MTE   : OP Type ∈ MTE_OPS（TransData / Transpose / Reshape / Copy 等）
        - Embedding: OP Type ∈ EMBEDDING_OPS（独立类别，计入 by_op_type，
                       但不并入四大聚合中——归到 other）
        - Other : 不在上述任何集合
        Embedding 单独保留 by_op_type 标签但在聚合层归入 other，因为
        Cube/Vector/MTE 修正系数体系不覆盖 embedding lookup。

    备注：
        - 该函数不依赖 step_trace 或目录约定，仅解析单个 CSV，便于
          外部脚本（roofline 校准、Timeloop 修正系数应用）直接调用。
        - 返回 fraction 字段当 total_us == 0 时统一返回 0.0（避免 ZeroDivision）。
        - 不修改任何全局状态。
    """
    csv_path = Path(op_summary_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"op_summary CSV not found: {csv_path}")

    rows_total = 0
    rows_counted = 0
    totals = {"cube": 0.0, "vector": 0.0, "mte": 0.0, "embedding": 0.0, "other": 0.0}
    by_op_type = defaultdict(lambda: {"category": "other", "count": 0, "total_us": 0.0})

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows_total += 1
            try:
                dur = float(r.get("Task Duration(us)", 0))
            except (TypeError, ValueError):
                continue
            if dur <= 0:
                continue
            op_type = r.get("OP Type", "") or ""
            op_name = r.get("Op Name", "") or ""
            cls = classify_op(op_type, op_name)
            totals[cls] = totals.get(cls, 0.0) + dur
            entry = by_op_type[op_type]
            entry["category"] = cls
            entry["count"] += 1
            entry["total_us"] += dur
            rows_counted += 1

    # Embedding 归入 other 进行最终聚合（保留 by_op_type 中的原 category 标签）
    total_cube = totals.get("cube", 0.0)
    total_vector = totals.get("vector", 0.0)
    total_mte = totals.get("mte", 0.0)
    total_other = totals.get("other", 0.0) + totals.get("embedding", 0.0)
    total_us = total_cube + total_vector + total_mte + total_other

    def frac(part):
        return round(part / total_us, 6) if total_us > 0 else 0.0

    by_op_type_out = {}
    for op_type, entry in by_op_type.items():
        cnt = entry["count"]
        tot = entry["total_us"]
        by_op_type_out[op_type] = {
            "category": entry["category"],
            "count": cnt,
            "total_us": round(tot, 4),
            "avg_us": round(tot / cnt, 4) if cnt > 0 else 0.0,
        }

    return {
        "source_csv": str(csv_path),
        "rows_total": rows_total,
        "rows_counted": rows_counted,
        "total_cube_us": round(total_cube, 4),
        "total_vector_us": round(total_vector, 4),
        "total_mte_us": round(total_mte, 4),
        "total_other_us": round(total_other, 4),
        "total_us": round(total_us, 4),
        "cube_fraction": frac(total_cube),
        "vector_fraction": frac(total_vector),
        "mte_fraction": frac(total_mte),
        "other_fraction": frac(total_other),
        "by_op_type": by_op_type_out,
    }


def parse_all():
    configs = [
        ("bert_base", 1, ["PipeUtilization", "L2Cache", "Memory", "ArithmeticUtilization"]),
        ("gpt2_small", 1, ["PipeUtilization", "L2Cache"]),
        ("qwen3_06b", 1, ["PipeUtilization", "Memory"]),
        ("bert_base", 16, ["Memory"]),
        ("gpt2_small", 16, ["Memory"]),
    ]
    all_results = {}
    for model, batch, metrics in configs:
        all_results[f"{model}_b{batch}"] = {}
        for metric in metrics:
            r = parse_model_metric(model, batch, metric)
            if r:
                all_results[f"{model}_b{batch}"][metric] = r
                print(f"✓ {model} b={batch} {metric}: wall={r['wall_clock']['iter_avg_ms']:.3f}ms "
                      f"npu={r['npu_op_sum']['per_iter_ms']:.3f}ms "
                      f"host={r['host_overhead']['fraction_of_wall']*100:.1f}%")
            else:
                print(f"✗ {model} b={batch} {metric} MISSING")

    out_path = MSPROF_DIR / "all_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--model", type=str)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--metric", type=str, default="PipeUtilization")
    p.add_argument("--msprof-base", type=str, default=None,
                   help="Override the msprof_data root (also settable via MSPROF_BASE env). "
                        "Default: <repo>/msprof_data")
    p.add_argument("--vector-mte-csv", type=str, default=None,
                   help="op_summary.csv path; runs extract_vector_mte_decomposition()")
    p.add_argument("--vector-mte-json", type=str, default=None,
                   help="output path for vector/MTE decomposition JSON. "
                        "If --vector-mte-csv is set, writes the decomposition; "
                        "otherwise prints to stdout.")
    args = p.parse_args()

    # CLI override of the msprof_data root (find_prof_dir / parse_all read the module global).
    if args.msprof_base:
        MSPROF_DIR = resolve_msprof_dir(args.msprof_base)

    # 优先处理新的 Cube/Vector/MTE 分解通路（与 main parse_all 互不影响）
    if args.vector_mte_csv:
        decomp = extract_vector_mte_decomposition(args.vector_mte_csv)
        out_text = json.dumps(decomp, indent=2, ensure_ascii=False)
        if args.vector_mte_json:
            out_path = Path(args.vector_mte_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(out_text)
            print(f"✓ vector/MTE decomposition written to {out_path}")
            print(f"  cube={decomp['total_cube_us']:.2f}us "
                  f"vector={decomp['total_vector_us']:.2f}us "
                  f"mte={decomp['total_mte_us']:.2f}us "
                  f"other={decomp['total_other_us']:.2f}us")
        else:
            print(out_text)
    elif args.all:
        parse_all()
    elif args.model:
        r = parse_model_metric(args.model, args.batch, args.metric)
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        parse_all()
