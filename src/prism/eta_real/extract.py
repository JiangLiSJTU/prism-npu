#!/usr/bin/env python3
"""
Phase M：从 msprof op_summary CSV 提取 Cube 利用率 + 算子级 shape 信息。

输入：msprof PipeUtilization / ArithmeticUtilization 等目录里的 op_summary_*.csv
输出：data/cube_util_extracted.json — 按 (model, S, B, op_type) 聚合的 cube_util_real

关键字段：
  - OP Type: 仅保留 MatMul / BatchMatMul / BatchMatMulV2 等 GEMM 算子
  - Input Shapes: 提 M, N, K (e.g. "1,512,1024;1,1024,3072;" → M=512, K=1024, N=3072)
  - aicore_time(us): NPU AICore 实际占用时间
  - aic_total_cycles: AIC 总周期
  - aic_mac_fp16_ratio: Cube MAC FP16 占比（= cube 利用率核心指标）
  - aic_mac_int8_ratio: Cube MAC INT8 占比
  - Task Duration(us): 算子总耗时

聚合：
  cube_util_weighted = Σ(aicore_time × aic_mac_fp16_ratio) / Σ(aicore_time)
                       仅按 GEMM 算子聚合
"""

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from collections import defaultdict


logger = logging.getLogger(__name__)

GEMM_OP_TYPES = {"MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"}


def parse_shape_string(shape_str):
    """Parse Input Shapes string from msprof CSV.

    Format examples (CSV-escaped quotes):
      A) raw form: 1,128,768;1,768,768;
      B) double-quoted: triple-doublequote separated dims
    Returns list of [dim1, dim2, ...].
    """
    if not shape_str:
        return []
    s = shape_str.strip()
    # 去掉所有双引号
    s = s.replace('"', '')
    if not s:
        return []
    parts = [p.strip() for p in s.split(';') if p.strip()]
    shapes = []
    for p in parts:
        try:
            dims = [int(x.strip()) for x in p.split(',') if x.strip().lstrip('-').isdigit()]
            if dims:
                shapes.append(dims)
        except ValueError:
            continue
    return shapes


def derive_mnk_from_matmul_shapes(shapes: list) -> tuple:
    """
    从 MatMul 的 Input Shapes 推导 (M, N, K)。
    标准 MatMul: A[..., M, K] × B[..., K, N] → C[..., M, N]
    BatchMatMul: 多了 batch 维。
    """
    if len(shapes) < 2:
        return (None, None, None)
    a_shape = shapes[0]
    b_shape = shapes[1]
    if len(a_shape) < 2 or len(b_shape) < 2:
        return (None, None, None)
    M = a_shape[-2]
    K = a_shape[-1]
    K2 = b_shape[-2]
    N = b_shape[-1]
    if K != K2:
        # MatMul 可能转置（adj）；尝试调整
        if a_shape[-1] == b_shape[-1]:
            # A 转置：A[K, M] × B[K, N] → 需要 swap
            M, K = a_shape[-1], a_shape[-2]
            N = b_shape[-1]
            K2 = b_shape[-2]
        if K != K2:
            return (None, None, None)
    return (M, N, K)


def _row_get_float(row: dict, key: str) -> float:
    """Safe float coercion for a CSV row field; returns 0.0 on missing/invalid."""
    try:
        return float(row.get(key, '') or 0)
    except (ValueError, TypeError):
        return 0.0


def parse_op_summary(csv_path: Path, gemm_only=True) -> list:
    """解析 op_summary CSV，返回算子记录列表。

    若 gemm_only=False，返回所有算子（含 Vector/MTE/Memcpy 等）——
    用于 Vector 占比分析。
    """
    records = []
    try:
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                op_type = (row.get('OP Type') or '').strip()
                if gemm_only and op_type not in GEMM_OP_TYPES:
                    continue
                op_name = (row.get('Op Name') or '').strip()
                input_shapes = row.get('Input Shapes', '').strip()
                shapes = parse_shape_string(input_shapes)
                M, N, K = derive_mnk_from_matmul_shapes(shapes)

                # ftry is now module-level _row_get_float to avoid per-row redefinition
                def ftry(key, _row=row):
                    return _row_get_float(_row, key)

                task_type = (row.get('Task Type') or '').strip()
                rec = {
                    'op_name': op_name,
                    'op_type': op_type,
                    'task_type': task_type,
                    'M': M, 'N': N, 'K': K,
                    'task_duration_us': ftry('Task Duration(us)'),
                    'aicore_time_us': ftry('aicore_time(us)'),
                    'aic_total_cycles': ftry('aic_total_cycles'),
                    'aic_mac_fp16_ratio': ftry('aic_mac_fp16_ratio'),
                    'aic_mac_int8_ratio': ftry('aic_mac_int8_ratio'),
                    'aic_cube_fops': ftry('aic_cube_fops'),
                    # AIV (Vector) 字段
                    'aiv_time_us': ftry('aiv_time(us)'),
                    'aiv_total_cycles': ftry('aiv_total_cycles'),
                    'aiv_vec_fp16_ratio': ftry('aiv_vec_fp16_ratio'),
                    'aiv_vec_fp32_ratio': ftry('aiv_vec_fp32_ratio'),
                    'aiv_vec_int32_ratio': ftry('aiv_vec_int32_ratio'),
                    'aiv_vec_misc_ratio': ftry('aiv_vec_misc_ratio'),
                    'aiv_vector_fops': ftry('aiv_vector_fops'),
                    'block_dim': int(row.get('Block Dim', 0) or 0),
                    'input_shapes_raw': input_shapes,
                }
                records.append(rec)
    except (csv.Error, OSError, ValueError) as e:
        logger.error("解析失败 %s: %s", csv_path, e)
        raise RuntimeError(f"failed to parse msprof CSV {csv_path}") from e
    return records


def aggregate_full_pipe(csv_path: Path) -> dict:
    """从完整 op_summary 提取 Cube/Vector/MTE 全管线占比。"""
    all_recs = parse_op_summary(csv_path, gemm_only=False)
    if not all_recs:
        return {}

    # 按 task_type 分类
    by_task_type = defaultdict(lambda: {'total_duration_us': 0.0, 'count': 0})
    total_duration = 0.0
    cube_active_us = 0.0
    vector_active_us = 0.0

    for r in all_recs:
        d = r['task_duration_us']
        tt = r['task_type'] or 'OTHER'
        by_task_type[tt]['total_duration_us'] += d
        by_task_type[tt]['count'] += 1
        total_duration += d
        # Cube active 时间 = aicore_time × mac_ratio
        cube_active_us += r['aicore_time_us'] * (r['aic_mac_fp16_ratio'] + r['aic_mac_int8_ratio'])
        # Vector active 时间 = aiv_time × vec_ratio
        vec_ratio_total = (r['aiv_vec_fp16_ratio'] + r['aiv_vec_fp32_ratio']
                           + r['aiv_vec_int32_ratio'] + r['aiv_vec_misc_ratio'])
        vector_active_us += r['aiv_time_us'] * vec_ratio_total

    cube_frac = cube_active_us / total_duration if total_duration > 0 else 0
    vector_frac = vector_active_us / total_duration if total_duration > 0 else 0

    return {
        'total_duration_us': round(total_duration, 3),
        'cube_active_us': round(cube_active_us, 3),
        'vector_active_us': round(vector_active_us, 3),
        'cube_active_pct_of_total': round(cube_frac * 100, 3),
        'vector_active_pct_of_total': round(vector_frac * 100, 3),
        'task_type_breakdown': {
            tt: {**v, 'total_duration_us': round(v['total_duration_us'], 3),
                 'pct': round(100 * v['total_duration_us'] / total_duration, 3) if total_duration else 0}
            for tt, v in by_task_type.items()
        },
    }


def aggregate_cube_util(records: list) -> dict:
    """聚合 cube_util_weighted。"""
    if not records:
        return {'n_ops': 0, 'cube_util_weighted_pct': 0.0}

    total_aicore_time = sum(r['aicore_time_us'] for r in records)
    weighted_fp16 = sum(r['aicore_time_us'] * r['aic_mac_fp16_ratio'] for r in records)
    weighted_int8 = sum(r['aicore_time_us'] * r['aic_mac_int8_ratio'] for r in records)

    cube_util_fp16 = weighted_fp16 / total_aicore_time if total_aicore_time > 0 else 0
    cube_util_int8 = weighted_int8 / total_aicore_time if total_aicore_time > 0 else 0
    cube_util_total = cube_util_fp16 + cube_util_int8  # 通常一种主导

    avg_block_dim = sum(r['block_dim'] for r in records) / len(records)

    return {
        'n_ops': len(records),
        'total_aicore_time_us': round(total_aicore_time, 3),
        'cube_util_fp16_pct': round(cube_util_fp16 * 100, 3),   # 与 cube_util_avg_pct 相符
        'cube_util_int8_pct': round(cube_util_int8 * 100, 3),
        'cube_util_total_pct': round(cube_util_total * 100, 3),
        'avg_block_dim': round(avg_block_dim, 1),
    }


def shape_distribution(records: list) -> list:
    """按 (M, N, K, op_type) 聚合算子统计（区分 BatchMatMul vs MatMul，影响 B× 缩放）。"""
    by_mnk = defaultdict(lambda: {
        'count': 0, 'aicore_time_us': 0.0,
        'sum_fp16_ratio_weighted': 0.0,
        'op_type': None,
    })
    for r in records:
        if r['M'] is None or r['N'] is None or r['K'] is None:
            continue
        # 区分 BatchMatMul vs MatMul：影响 M 是否 per-batch
        is_batch = 'BatchMatMul' in r['op_type']
        key = (r['M'], r['N'], r['K'], 'BMM' if is_batch else 'MM')
        by_mnk[key]['count'] += 1
        by_mnk[key]['aicore_time_us'] += r['aicore_time_us']
        by_mnk[key]['sum_fp16_ratio_weighted'] += r['aicore_time_us'] * r['aic_mac_fp16_ratio']
        by_mnk[key]['op_type'] = r['op_type']

    result = []
    for (M, N, K, kind), agg in sorted(by_mnk.items(), key=lambda x: -x[1]['aicore_time_us']):
        cube_u = agg['sum_fp16_ratio_weighted'] / agg['aicore_time_us'] if agg['aicore_time_us'] > 0 else 0
        result.append({
            'M': M, 'N': N, 'K': K,
            'op_kind': kind,   # 'BMM' = BatchMatMul (M per-batch), 'MM' = MatMul (M includes batch)
            'op_type': agg['op_type'],
            'count': agg['count'],
            'aicore_time_us': round(agg['aicore_time_us'], 3),
            'cube_util_pct': round(cube_u * 100, 3),
        })
    return result


def find_op_summary_csv(prof_dir: Path) -> Path:
    """在 msprof 输出目录里找 op_summary CSV。"""
    matches = list(prof_dir.rglob('op_summary*.csv'))
    if not matches:
        return None
    # 选最新（按 mtime 或 filename timestamp）
    return max(matches, key=lambda p: p.stat().st_mtime)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--msprof-data-dir', default='./msprof_data',
                   help='msprof_data 根目录')
    p.add_argument('--output', default='./data/cube_util_extracted.json',
                   help='输出 JSON 路径')
    p.add_argument('--prefixes', nargs='+',
                   default=['msprof_qwen3_06b', 'msprof_qwen3_prefill', 'msprof_bert_base', 'msprof_gpt2_small'],
                   help='msprof 目录前缀过滤')
    args = p.parse_args()

    msprof_root = Path(args.msprof_data_dir)
    if not msprof_root.exists():
        print(f"ERROR: {msprof_root} 不存在")
        return 1

    results = {}
    for d in sorted(msprof_root.iterdir()):
        if not d.is_dir():
            continue
        if not any(d.name.startswith(pre) for pre in args.prefixes):
            continue

        csv_path = find_op_summary_csv(d)
        if csv_path is None:
            print(f"  跳过 {d.name}（无 op_summary CSV）")
            continue

        records = parse_op_summary(csv_path)
        agg = aggregate_cube_util(records)
        shapes = shape_distribution(records)
        full_pipe = aggregate_full_pipe(csv_path)

        results[d.name] = {
            'csv_path': str(csv_path.relative_to(msprof_root)),
            'agg': agg,
            'full_pipe': full_pipe,
            'top_shapes_by_aicore_time': shapes[:10],
        }
        cube_pct_total = full_pipe.get('cube_active_pct_of_total', 0)
        vec_pct_total = full_pipe.get('vector_active_pct_of_total', 0)
        print(f"  ✓ {d.name}: GEMM cube_util={agg['cube_util_total_pct']:.2f}%, "
              f"全管线 cube={cube_pct_total:.2f}%, vector={vec_pct_total:.2f}%, "
              f"GEMM AIcore={agg['total_aicore_time_us']:.1f} μs")

    # 输出
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== 写入 {out_path}（{len(results)} 条记录）===")

    # 终端汇总表
    print("\n=== Cube 利用率汇总（按目录名）===")
    print(f"{'msprof dir':<55} {'n_ops':>6} {'cube_util_fp16%':>16} {'aicore_us':>12}")
    for name, data in sorted(results.items()):
        a = data['agg']
        print(f"{name:<55} {a['n_ops']:>6} {a['cube_util_fp16_pct']:>16.2f} {a['total_aicore_time_us']:>12.1f}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
