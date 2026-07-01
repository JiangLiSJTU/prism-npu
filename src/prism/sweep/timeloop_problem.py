#!/usr/bin/env python3
"""
timeloop_sweep_v3.py — Real Timeloop GEMM Sweep via Docker
===========================================================
Phase F4/F5：用真实 timeloop-mapper 替代 Phase D 解析模型扫描

Docker 镜像（本地 Mac ARM64）：
  timeloopaccelergy/accelergy-timeloop-infrastructure:latest-arm64

使用方法：
  # F1 校准（BERT-base 8 GEMM，单基线配置）
  python3 timeloop_sweep_v3.py --chip 910b4 --model bert_base --f1

  # F4 粗扫（910B4，36 配置）
  python3 timeloop_sweep_v3.py --chip 910b4 --model bert_base --coarse

  # F5 310P 粗扫
  python3 timeloop_sweep_v3.py --chip 310p --model bert_base --coarse

  # Dry run（验证 YAML 生成，不实际调用 Docker）
  python3 timeloop_sweep_v3.py --chip 910b4 --model bert_base --f1 --dry-run

版本历史：
  v3.0  2026-05-06  Phase F4/F5 初始版本（真实 Timeloop Docker 调用）
                    修复：--entrypoint /bin/bash 绕过 s6-init ARM64 兼容问题
                    修复：problems: → problem: 单数格式转换
                    修复：DOCKER_IMAGE 使用 latest-arm64 标签
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ── 路径配置 ─────────────────────────────────────────────────────────────
# Resolve repo root robustly across editable install, pip install, and ad-hoc cwd.
# Override via PRISM_ROOT env var.
from prism import data_root_or_fallback

SCRIPT_DIR  = Path(__file__).parent.resolve()
REPO_ROOT   = data_root_or_fallback()
ARCH_DIR    = REPO_ROOT / "arch"
MAPPER_DIR  = REPO_ROOT / "mapper"
MODELS_DIR  = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "timeloop_results"

# ── Docker 配置（本地 Mac ARM64）──────────────────────────────────────────
DOCKER_IMAGE = "timeloopaccelergy/accelergy-timeloop-infrastructure:latest-arm64"

# ── 芯片基线规格 ──────────────────────────────────────────────────────────
CHIP_SPECS = {
    "910b4": {
        "arch_yaml":        ARCH_DIR / "ascend_910b4.yaml",
        "constraints_yaml": MAPPER_DIR / "timeloop_constraints_910b4.yaml",
        "clock_hz":         1_500_000_000,
        "fp16_tflops":      280.0,
        "hbm_bw_gbs":       392.0,
        "l2_mb":            192.0,
        "cube_mac_per_core": 4096,
        "num_cores":        24,
        # 基线 L2 depth（bytes）：192MB = 25165824 bytes (depth at width=8,word-bits=8)
        # depth × width × word-bits / 8 = depth × 1 byte
        "l2_depth_base":    25165824,
        "bw_base":          392000,   # MB/s（与 YAML 中 bandwidth 字段一致）
        # Cube_MAC 范围：[0..4095] → split: 4095
        "cube_mac_count":   4096,
        "est_area_mm2":     50.0,
        "est_power_w":      310.0,
    },
    "310p": {
        "arch_yaml":        ARCH_DIR / "ascend_310p_timeloop.yaml",
        "constraints_yaml": MAPPER_DIR / "timeloop_constraints_310p.yaml",
        "clock_hz":         1_000_000_000,
        "fp16_tflops":      70.0,
        "hbm_bw_gbs":       204.8,
        "l2_mb":            8.0,
        "cube_mac_per_core": 4096,
        "num_cores":        8,
        # 310P L2: arch YAML depth=524288 (CACTI 近似 1MB，实际 8MB)
        # l2_depth_base 须与 YAML depth 完全匹配，供 scale_arch_yaml_text 正则替换
        "l2_depth_base":    524288,
        "bw_base":          204800,   # MB/s（与 YAML bandwidth: 204800 一致）
        "cube_mac_count":   4096,
        "est_area_mm2":     15.0,
        "est_power_w":      67.0,
    },
}

# ── 扫描空间定义 ──────────────────────────────────────────────────────────
SWEEP_SPACES = {
    "coarse": {
        # 粗扫：3×4×3 = 36 配置
        "cube_mac_scale": [1.0, 2.0, 4.0],
        "l2_scale":       [0.5, 1.0, 2.0, 4.0],
        "bw_scale":       [0.5, 1.0, 2.0],
    },
    "fine": {
        # 细扫：5×5×4 = 100 配置（围绕 sweet spot）
        "cube_mac_scale": [1.0, 1.5, 2.0, 3.0, 4.0],
        "l2_scale":       [0.5, 1.0, 2.0, 4.0, 8.0],
        "bw_scale":       [0.5, 1.0, 1.5, 2.0],
    },
    "full": {
        # 完整扫描：4×5×4 = 80 配置
        "cube_mac_scale": [1.0, 2.0, 4.0, 8.0],
        "l2_scale":       [0.25, 0.5, 1.0, 2.0, 4.0],
        "bw_scale":       [0.5, 1.0, 1.5, 2.0],
    },
}


# ────────────────────────────────────────────────────────────────────────────
# YAML 工具函数
# ────────────────────────────────────────────────────────────────────────────

def load_model_ops(model_yaml_path: Path) -> list:
    """
    加载 model YAML 中的 problems 列表。
    支持 problems: (list) 格式（我们的内部格式）。
    """
    with open(model_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "problems" in data:
        return data["problems"]
    elif "problem" in data:
        return [data["problem"]]
    else:
        raise ValueError(f"model YAML 缺少 'problems' 或 'problem' 字段: {model_yaml_path}")


def _fix_projection(proj):
    """
    将我们的投影格式转换为 Timeloop 正确的三层嵌套格式。

    我们的格式（来自 bert_base.yaml）：
      [[K], [N]]  → Python: [["K"], ["N"]]
      每个元素是 factor_group，元素内是维度名字符串

    Timeloop 期望格式（官方 gemm_ABZ.yaml）：
      [ [[K]], [[N]] ]  → Python: [[["K"]], [["N"]]]
      每个元素是 factor_group（list of terms），每个 term 是 [dim] 数组

    断言：term.isArray() 要求 term 是数组（list），不是标量（string）
    """
    result = []
    for factor_group in proj:
        if isinstance(factor_group, list):
            new_group = []
            for term in factor_group:
                if isinstance(term, str):
                    # 字符串 "K" → 包装为 term 数组 ["K"]
                    new_group.append([term])
                elif isinstance(term, list):
                    # 已经是 list（如 ["K", 2]）→ 保持
                    new_group.append(term)
                else:
                    new_group.append([str(term)])
            result.append(new_group)
        else:
            # 标量直接包装
            result.append([[str(factor_group)]])
    return result


def _fix_shape_keywords(shape: dict) -> dict:
    """
    修复 shape 中的连字符关键词 → 下划线（Timeloop 新版要求）。
    data-spaces → data_spaces
    read-write  → read_write
    并修复每个 data_space 的 projection 格式（三层嵌套）。
    """
    new_shape = {}
    for k, v in shape.items():
        new_key = k.replace("-", "_")
        if new_key == "data_spaces" and isinstance(v, list):
            new_ds = []
            for ds in v:
                new_ds_entry = {}
                for dk, dv in ds.items():
                    new_dk = dk.replace("-", "_")
                    if new_dk == "projection" and isinstance(dv, list):
                        new_ds_entry[new_dk] = _fix_projection(dv)
                    else:
                        new_ds_entry[new_dk] = dv
                new_ds.append(new_ds_entry)
            new_shape[new_key] = new_ds
        else:
            new_shape[new_key] = v
    return new_shape


def convert_op_to_timeloop_problem(op: dict) -> dict:
    """
    将我们的 problems: list 中的单个 op 转换为 Timeloop 期望的 problem: 格式。

    输入（我们的格式）：
      name: q_projection
      shape: {name: GEMM, dimensions: [...], data-spaces: [...]}
      instance: {M: 128, N: 768, K: 768}

    输出（Timeloop 格式，修复关键词和 projection 嵌套）：
      problem:
        shape:    {data_spaces: [...], dimensions: [...], name: GEMM}
        instance: {M: 128, N: 768, K: 768}

    关键修复：
      - data-spaces → data_spaces
      - read-write  → read_write
      - projection [[K],[N]] → [[[K]],[[N]]] （term 须为 array）
    """
    raw_shape = op.get("shape", {})
    fixed_shape = _fix_shape_keywords(raw_shape)
    return {
        "problem": {
            "shape":    fixed_shape,
            "instance": op.get("instance", {}),
        }
    }


def scale_arch_yaml_text(arch_text: str, chip: str,
                          cube_mac_scale: float,
                          l2_scale: float,
                          bw_scale: float) -> str:
    """
    在 arch YAML 文本中替换关键参数：
      - L2 depth（字节数）
      - DRAM bandwidth
      - Cube_MAC 数组大小（[0..N]）

    Implementation note: uses ``re.sub`` on the YAML *text* (not a structured
    yaml load → mutate → dump). This is **deliberate** — Timeloop's parser is
    sensitive to YAML key ordering, anchor / alias placement, and explicit
    integer vs. float typing that ``yaml.safe_dump`` would normalize away.
    Regex preserves the exact original formatting and only swaps the numeric
    literals matching ``CHIP_SPECS[chip]['*_base']`` values.

    Constraint: input MUST come from a known arch yaml whose ``*_base`` literals
    are unique within the file. New chips must be vetted manually.
    """
    spec = CHIP_SPECS[chip]

    # 1. L2 depth 替换
    new_l2_depth = int(spec["l2_depth_base"] * l2_scale)
    arch_text = re.sub(
        r"(depth:\s*)" + str(spec["l2_depth_base"]),
        r"\g<1>" + str(new_l2_depth),
        arch_text
    )

    # 2. DRAM bandwidth 替换（只替换 DRAM 节下的）
    new_bw = int(spec["bw_base"] * bw_scale)
    arch_text = re.sub(
        r"(bandwidth:\s*)" + str(spec["bw_base"]),
        r"\g<1>" + str(new_bw),
        arch_text
    )

    # 3. Cube_MAC 数组大小替换
    orig_count = spec["cube_mac_count"]
    new_count  = int(orig_count * cube_mac_scale)
    orig_range = f"0..{orig_count - 1}"
    new_range  = f"0..{new_count - 1}"
    arch_text  = arch_text.replace(
        f"Cube_MAC[{orig_range}]",
        f"Cube_MAC[{new_range}]"
    )

    return arch_text


def make_f1_mapper_yaml() -> str:
    """
    生成用于 F1 校准的精简 mapper 配置（快速搜索，不追求最优）。
    """
    return """mapper:
  id-config: exhaustive
  search-size: 500
  num-threads: 4
  timeout: 55
  victory-condition: 500
  optimization-metric: [delay]
  algorithm: random-pruned
  random-seed: 42
  out-prefix: timeloop-mapper
  mapspace:
    template: uber
    split-permutations: true
    global:
      max-parallelism-per-dim: 64
"""


# ────────────────────────────────────────────────────────────────────────────
# Timeloop Docker 调用
# ────────────────────────────────────────────────────────────────────────────

def run_timeloop_op(
    op_name: str,
    problem_yaml_text: str,
    arch_yaml_text: str,
    mapper_yaml_text: str,
    constraints_yaml_path: Path,
    output_dir: Path,
    timeout: int = 120,
    dry_run: bool = False,
) -> dict:
    """
    对单个 GEMM 算子调用 timeloop-mapper Docker 容器。

    目录结构：
      output_dir/
        inputs/
          arch.yaml
          mapper.yaml
          constraints.yaml
          model.yaml       ← 单个 problem
        (timeloop 输出写入 output_dir/)
    """
    op_dir     = output_dir / op_name
    inputs_dir = op_dir / "inputs"
    op_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # 写入 YAML 文件
    (inputs_dir / "arch.yaml").write_text(arch_yaml_text, encoding="utf-8")
    (inputs_dir / "mapper.yaml").write_text(mapper_yaml_text, encoding="utf-8")
    shutil.copy(constraints_yaml_path, inputs_dir / "constraints.yaml")
    (inputs_dir / "model.yaml").write_text(problem_yaml_text, encoding="utf-8")

    if dry_run:
        print(f"  [dry-run] {op_name}: YAML 写入完成，跳过 Docker 调用")
        return {"op": op_name, "success": False, "dry_run": True}

    # ── Docker 命令 ──────────────────────────────────────────────────────
    # 关键：--entrypoint /bin/bash 绕过 s6-init（x86 only）在 ARM64 上的问题
    # 必须传入 compound 组件 YAML，否则 Accelergy ERT 为空 → timeloop-mapper crash
    _COMP = (
        "/usr/local/src/timeloop-python/tests/timeloop-accelergy-exercises"
        "/workspace/example_designs/example_designs/_components"
    )
    timeloop_cmd = (
        "cd /home/workspace/run && "
        "timeloop-mapper "
        "/home/workspace/inputs/arch.yaml "
        "/home/workspace/inputs/mapper.yaml "
        "/home/workspace/inputs/constraints.yaml "
        "/home/workspace/inputs/model.yaml "
        f"{_COMP}/smartbuffer_SRAM.yaml "
        f"{_COMP}/intmac.yaml "
        f"{_COMP}/regfile.yaml "
        "2>&1"
    )
    docker_cmd = [
        "docker", "run", "--rm",
        "--entrypoint", "/bin/bash",
        "-v", f"{op_dir.resolve()}:/home/workspace/run",
        "-v", f"{inputs_dir.resolve()}:/home/workspace/inputs:ro",
        DOCKER_IMAGE,
        "-c", timeloop_cmd,
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        success = (proc.returncode == 0)

        if not success:
            # 保存错误日志
            (op_dir / "error.log").write_text(
                f"returncode={proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}",
                encoding="utf-8"
            )
            return {
                "op":         op_name,
                "success":    False,
                "returncode": proc.returncode,
                "elapsed_s":  round(elapsed, 1),
                "stderr":     proc.stderr[:500],
            }

        # 解析 stats
        stats_file = op_dir / "timeloop-mapper.stats.txt"
        if stats_file.exists():
            stats = parse_stats_file(stats_file)
            return {
                "op":         op_name,
                "success":    True,
                "elapsed_s":  round(elapsed, 1),
                **stats,
            }
        else:
            return {
                "op":      op_name,
                "success": False,
                "error":   "stats.txt not generated (timeloop ran but no output)",
                "elapsed_s": round(elapsed, 1),
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {
            "op":      op_name,
            "success": False,
            "error":   f"Timeout after {timeout}s",
            "elapsed_s": round(elapsed, 1),
        }
    except Exception as e:
        return {
            "op":      op_name,
            "success": False,
            "error":   str(e),
            "elapsed_s": round(time.time() - t0, 1),
        }


# ────────────────────────────────────────────────────────────────────────────
# Stats 解析（内联版本，避免跨模块依赖）
# ────────────────────────────────────────────────────────────────────────────

def parse_stats_file(stats_path: Path) -> dict:
    """解析 timeloop-mapper.stats.txt，提取 cycles/energy/area/utilization。

    timeloop-mapper 实际输出格式（冒号分隔，非等号）：
      Cycles                  : 8192
      Energy (total)          : 67758981.12 pJ
      Area (total)            : 60907782.00 um^2   ← 单位 um^2，需转 mm^2
    Summary Stats 节（文件末尾）：
      Utilization: 9.38%
    """
    text = stats_path.read_text(encoding="utf-8", errors="replace")
    result = {
        "cycles":      None,
        "energy_pj":   None,
        "area_mm2":    None,
        "utilization": None,
        "gflops":      None,
    }

    # cycles：取第一个顶层 "Cycles  : N" 行
    m = re.search(r"^\s*Cycles\s*:\s*([0-9]+)", text, re.IGNORECASE | re.MULTILINE)
    if m:
        result["cycles"] = float(m.group(1))

    # energy：顶层 "Energy (total) : N pJ"
    m = re.search(r"^\s*Energy\s*\(total\)\s*:\s*([0-9.eE+\-]+)\s*pJ",
                  text, re.IGNORECASE | re.MULTILINE)
    if m:
        result["energy_pj"] = float(m.group(1))

    # area：顶层 "Area (total) : N um^2" → 转为 mm^2（÷ 1e6）
    m = re.search(r"^\s*Area\s*\(total\)\s*:\s*([0-9.eE+\-]+)\s*um\^2",
                  text, re.IGNORECASE | re.MULTILINE)
    if m:
        result["area_mm2"] = float(m.group(1)) / 1e6

    # utilization：Summary Stats 节 "Utilization: 9.38%"
    m = re.search(r"Utilization:\s*([0-9.]+)%", text, re.IGNORECASE)
    if m:
        result["utilization"] = float(m.group(1))

    return result


# ────────────────────────────────────────────────────────────────────────────
# 单次配置运行（遍历所有 GEMM 算子）
# ────────────────────────────────────────────────────────────────────────────

def run_one_config(
    chip: str,
    ops: list,
    config: dict,
    output_dir: Path,
    timeout: int = 120,
    dry_run: bool = False,
    f1_mode: bool = False,
) -> dict:
    """
    对一个 (cube_mac_scale, l2_scale, bw_scale) 配置运行所有算子。

    返回聚合结果：cycles_total, latency_gemm_us, energy_total_pj, ...
    """
    spec       = CHIP_SPECS[chip]
    arch_text  = spec["arch_yaml"].read_text(encoding="utf-8")
    const_path = spec["constraints_yaml"]

    # 缩放 arch
    arch_text_scaled = scale_arch_yaml_text(
        arch_text,
        chip,
        config["cube_mac_scale"],
        config["l2_scale"],
        config["bw_scale"],
    )

    # mapper YAML：F1 校准用精简版，正式扫描用完整版
    if f1_mode:
        mapper_text = make_f1_mapper_yaml()
    else:
        mapper_text = (MAPPER_DIR / "timeloop_mapper.yaml").read_text(encoding="utf-8")

    total_cycles    = 0
    total_energy_pj = 0.0
    max_area_mm2    = 0.0
    any_success     = False
    ops_results     = {}

    for op in ops:
        op_name     = op.get("name", "unknown")
        problem_dict = convert_op_to_timeloop_problem(op)
        problem_text = yaml.dump(problem_dict, default_flow_style=False, allow_unicode=True)

        op_output_dir = output_dir
        result = run_timeloop_op(
            op_name=op_name,
            problem_yaml_text=problem_text,
            arch_yaml_text=arch_text_scaled,
            mapper_yaml_text=mapper_text,
            constraints_yaml_path=const_path,
            output_dir=op_output_dir,
            timeout=timeout,
            dry_run=dry_run,
        )

        ops_results[op_name] = {
            "success":    result.get("success", False),
            "returncode": result.get("returncode"),
            "elapsed_s":  result.get("elapsed_s"),
        }

        if result.get("success"):
            any_success = True
            cycles    = result.get("cycles") or 0.0
            energy_pj = result.get("energy_pj") or 0.0
            area_mm2  = result.get("area_mm2") or 0.0
            total_cycles    += cycles
            total_energy_pj += energy_pj
            max_area_mm2     = max(max_area_mm2, area_mm2)
            ops_results[op_name]["cycles"]    = int(cycles)
            ops_results[op_name]["energy_pj"] = round(energy_pj, 2)
            ops_results[op_name]["area_mm2"]  = round(area_mm2, 4)
        else:
            if not dry_run:
                print(f"    ✗ {op_name}: {result.get('error', 'failed')}")

    # 延迟计算
    clock_hz        = spec["clock_hz"] * config["cube_mac_scale"]  # 算力正比于 MAC 数
    # 注意：clock_hz 这里代表等效算力，不是真实频率
    # 实际上 Timeloop cycles 已经考虑了 MAC 并行度，直接除以基线频率
    real_clock_hz   = spec["clock_hz"]
    latency_gemm_us = (total_cycles / real_clock_hz * 1e6) if total_cycles > 0 else None

    # 面积/功耗估算（线性缩放）
    cube_mac_scale  = config["cube_mac_scale"]
    l2_scale        = config["l2_scale"]
    est_area_mm2    = spec["est_area_mm2"] * (0.4 * cube_mac_scale + 0.4 * l2_scale + 0.2)
    est_power_w     = spec["est_power_w"]  * cube_mac_scale * 0.8
    effective_tflops = spec["fp16_tflops"] * cube_mac_scale
    tops_per_mm2    = effective_tflops * 1000 / est_area_mm2
    tops_per_w      = effective_tflops * 1000 / est_power_w

    return {
        "chip":                chip,
        "config":              config,
        "cube_mac_per_core":   int(spec["cube_mac_per_core"] * cube_mac_scale),
        "l2_mb":               round(spec["l2_mb"] * l2_scale, 1),
        "bw_gbs":              round(spec["hbm_bw_gbs"] * config["bw_scale"], 1),
        "fp16_tflops":         round(effective_tflops, 1),
        "cycles_total":        int(total_cycles) if any_success else None,
        "latency_gemm_us":     round(latency_gemm_us, 2) if latency_gemm_us else None,
        "latency_per_layer_us": round(latency_gemm_us / 12, 2) if latency_gemm_us else None,
        "energy_total_pj":     round(total_energy_pj, 2),
        "est_area_mm2":        round(est_area_mm2, 2),
        "est_power_w":         round(est_power_w, 1),
        "tops_per_mm2":        round(tops_per_mm2, 2),
        "tops_per_w":          round(tops_per_w, 3),
        "any_success":         any_success,
        "ops_results":         ops_results,
    }


# ────────────────────────────────────────────────────────────────────────────
# F1 校准入口
# ────────────────────────────────────────────────────────────────────────────

def run_f1_calibration(chip: str, model: str, timeout: int = 90,
                       dry_run: bool = False) -> dict:
    """
    F1 校准：单基线配置（所有 scale=1.0），BERT-base 8 GEMM 算子。
    目标：与 msprof 实测 NPU op time（BERT-base b1: 1.826ms）对比，偏差 ≤ 25%。
    """
    model_yaml = MODELS_DIR / f"{model}.yaml"
    if not model_yaml.exists():
        print(f"ERROR: model YAML not found: {model_yaml}", file=sys.stderr)
        sys.exit(1)

    ops = load_model_ops(model_yaml)
    print(f"\n{'='*60}")
    print(f"F1 校准：{model}（{chip}），{len(ops)} 个 GEMM 算子")
    print(f"{'='*60}")
    print(f"  Docker 镜像：{DOCKER_IMAGE}")
    print(f"  每算子超时：{timeout}s")
    if dry_run:
        print("  模式：dry-run（不调用 Docker）")

    config = {"cube_mac_scale": 1.0, "l2_scale": 1.0, "bw_scale": 1.0}
    cfg_name   = f"cm{config['cube_mac_scale']}_l2{config['l2_scale']}_bw{config['bw_scale']}"
    output_dir = RESULTS_DIR / f"{chip}_{model}_f1_calibration" / f"config_0000_{cfg_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_one_config(
        chip=chip,
        ops=ops,
        config=config,
        output_dir=output_dir,
        timeout=timeout,
        dry_run=dry_run,
        f1_mode=True,
    )
    result["config_idx"] = 0
    result["model"]      = model
    result["cfg_name"]   = cfg_name

    # 保存结果
    result_file = RESULTS_DIR / f"{chip}_{model}_f1_calibration" / "f1_result.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    # 报告
    print(f"\nF1 结果：")
    lat = result.get("latency_gemm_us")
    per_layer = result.get("latency_per_layer_us")
    print(f"  Timeloop GEMM 总延迟：{lat} μs")
    print(f"  Per-layer 延迟：{per_layer} μs/层")

    if lat is not None:
        # BERT-base b1 msprof anchor（12 层完整推理）
        # bert_base.yaml 定义 1 个 transformer block (8 GEMMs)
        # msprof 1.826ms = 12 层全程；按层对比需除以 12
        BERT_LAYERS      = 12
        msprof_npu_ms    = 1.826
        msprof_total_us  = msprof_npu_ms * 1000.0              # 1826 μs (12 层)
        msprof_cube_us   = msprof_total_us * 0.85              # 1552 μs Cube (12 层)
        msprof_cube_per_layer = msprof_cube_us / BERT_LAYERS   # 129.3 μs/层

        # Timeloop lat = 1 层的 GEMM 周期延迟（与 per_layer 对比）
        bias_vs_cube_pct = (lat - msprof_cube_per_layer) / msprof_cube_per_layer * 100
        # 12 层外推（用于与完整 msprof 对比）
        lat_12layer_us   = lat * BERT_LAYERS
        bias_12layer_pct = (lat_12layer_us - msprof_cube_us) / msprof_cube_us * 100
        passed = abs(bias_vs_cube_pct) <= 25.0
        status = "✅ PASS" if passed else "⚠ WARN"
        print(f"\n  msprof 对比（BERT-base b1 实测，12 层）：")
        print(f"    msprof NPU total（12层）：{msprof_total_us:.0f} μs")
        print(f"    msprof Cube（12层）：     {msprof_cube_us:.0f} μs（85% Cube）")
        print(f"    msprof Cube / 层：        {msprof_cube_per_layer:.1f} μs/层")
        print(f"  Timeloop（1层 8 GEMM）：")
        print(f"    Timeloop 1层 GEMM：       {lat:.1f} μs")
        print(f"    Timeloop 12层外推：        {lat_12layer_us:.1f} μs")
        print(f"  偏差（per-layer vs Cube）：  {bias_vs_cube_pct:+.1f}%  {status}（目标 ±25%）")
        print(f"  偏差（12层外推 vs Cube）：   {bias_12layer_pct:+.1f}%")
    else:
        failed_ops = [k for k, v in result.get("ops_results", {}).items()
                      if not v.get("success")]
        print(f"\n  ⚠ 无有效结果。失败算子：{failed_ops}")
        print(f"  检查各算子的 error.log：{output_dir}/")

    print(f"\n  结果保存至：{result_file}")
    return result


# ────────────────────────────────────────────────────────────────────────────
# 参数扫描主函数
# ────────────────────────────────────────────────────────────────────────────

def run_sweep(chip: str, model: str, sweep_type: str = "coarse",
              timeout: int = 120, dry_run: bool = False,
              num_threads: int = 1) -> list:
    """
    遍历扫描空间，对每个配置调用 run_one_config。
    """
    model_yaml = MODELS_DIR / f"{model}.yaml"
    if not model_yaml.exists():
        print(f"ERROR: model YAML not found: {model_yaml}", file=sys.stderr)
        sys.exit(1)
    ops = load_model_ops(model_yaml)

    space       = SWEEP_SPACES[sweep_type]
    configs     = list(product(
        space["cube_mac_scale"],
        space["l2_scale"],
        space["bw_scale"],
    ))
    total_cfgs  = len(configs)

    print(f"\n{'='*60}")
    print(f"Timeloop Sweep  chip={chip}  model={model}  type={sweep_type}")
    print(f"{'='*60}")
    print(f"  配置总数：{total_cfgs}（{len(space['cube_mac_scale'])} × "
          f"{len(space['l2_scale'])} × {len(space['bw_scale'])}）")
    print(f"  每算子超时：{timeout}s  并发：{num_threads}")
    if dry_run:
        print("  模式：dry-run")

    sweep_dir = RESULTS_DIR / f"{chip}_{model}_{sweep_type}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    def process_cfg(idx_cfg):
        idx, (cm, l2, bw) = idx_cfg
        config   = {"cube_mac_scale": cm, "l2_scale": l2, "bw_scale": bw}
        cfg_name = f"cm{cm}_l2{l2}_bw{bw}"
        out_dir  = sweep_dir / f"config_{idx:04d}_{cfg_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{idx+1:3d}/{total_cfgs}] {cfg_name} ...", end="", flush=True)
        result        = run_one_config(chip, ops, config, out_dir, timeout, dry_run)
        result["config_idx"] = idx
        result["cfg_name"]   = cfg_name
        result["model"]      = model
        status = "✓" if result["any_success"] else "✗"
        lat    = result.get("latency_gemm_us")
        print(f" {status}  {lat} μs" if lat else f" {status}")
        return result

    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = {ex.submit(process_cfg, (i, cfg)): i
                       for i, cfg in enumerate(configs)}
            for fut in as_completed(futures):
                all_results.append(fut.result())
        all_results.sort(key=lambda r: r["config_idx"])
    else:
        for i, cfg in enumerate(configs):
            all_results.append(process_cfg((i, cfg)))

    # 保存完整结果
    sweep_json = sweep_dir / "sweep_all.json"
    sweep_json.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )
    print(f"\n  完整结果保存至：{sweep_json}")

    # Pareto 分析
    valid   = [r for r in all_results if r.get("latency_gemm_us") is not None]
    print(f"  有效配置：{len(valid)} / {total_cfgs}")

    if valid:
        pareto = compute_pareto(valid)
        pareto_json = sweep_dir / "pareto_front.json"
        pareto_json.write_text(
            json.dumps(pareto, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8"
        )
        print(f"  Pareto 非支配点：{len(pareto)}，保存至 {pareto_json}")

        sweet = find_sweet_spot(pareto)
        if sweet:
            sweet_json = sweep_dir / "sweet_spot.json"
            sweet_json.write_text(
                json.dumps(sweet, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8"
            )
            print_sweet_spot(sweet)

    return all_results


# ────────────────────────────────────────────────────────────────────────────
# Pareto 分析
# ────────────────────────────────────────────────────────────────────────────

def compute_pareto(results: list) -> list:
    """
    计算 (latency, area) 二维 Pareto 前沿（最小化两者）。
    """
    pareto = []
    for r in results:
        lat  = r.get("latency_gemm_us", float("inf"))
        area = r.get("est_area_mm2", float("inf"))
        dominated = False
        for other in results:
            o_lat  = other.get("latency_gemm_us", float("inf"))
            o_area = other.get("est_area_mm2", float("inf"))
            if (o_lat <= lat and o_area <= area and
                    (o_lat < lat or o_area < area)):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    return sorted(pareto, key=lambda r: r.get("latency_gemm_us", float("inf")))


def find_sweet_spot(pareto: list) -> Optional[dict]:
    """
    在 Pareto 前沿中，用加权评分找 sweet spot：
      score = 0.6 × (lat_norm) + 0.4 × (area_norm)
    越小越好。
    """
    if not pareto:
        return None
    lats  = [r.get("latency_gemm_us", float("inf")) for r in pareto]
    areas = [r.get("est_area_mm2",    float("inf")) for r in pareto]
    lat_min, lat_max   = min(lats),  max(lats)
    area_min, area_max = min(areas), max(areas)

    best_score = float("inf")
    best       = None
    for r, lat, area in zip(pareto, lats, areas):
        lat_n  = (lat  - lat_min)  / (lat_max  - lat_min  + 1e-9)
        area_n = (area - area_min) / (area_max - area_min + 1e-9)
        score  = 0.6 * lat_n + 0.4 * area_n
        if score < best_score:
            best_score = score
            best       = r
    if best:
        best = dict(best)
        best["sweet_spot_score"] = round(best_score, 4)
    return best


def print_sweet_spot(sweet: dict) -> None:
    cfg = sweet.get("config", {})
    print(f"\n  ★ Sweet Spot 推荐配置：")
    print(f"    Cube-MAC scale: {cfg.get('cube_mac_scale')}x  "
          f"L2 scale: {cfg.get('l2_scale')}x  "
          f"BW scale: {cfg.get('bw_scale')}x")
    print(f"    FP16 TFLOPS: {sweet.get('fp16_tflops')}")
    print(f"    L2: {sweet.get('l2_mb')} MB  BW: {sweet.get('bw_gbs')} GB/s")
    print(f"    Latency: {sweet.get('latency_gemm_us')} μs  "
          f"Area: {sweet.get('est_area_mm2')} mm²")
    print(f"    TOPS/mm²: {sweet.get('tops_per_mm2')}  "
          f"TOPS/W: {sweet.get('tops_per_w')}")


# ────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="timeloop_sweep_v3.py — 真实 Timeloop Docker 扫描"
    )
    parser.add_argument("--chip", default="910b4",
                        choices=list(CHIP_SPECS.keys()),
                        help="芯片型号")
    parser.add_argument("--model", default="bert_base",
                        help="模型名称（对应 models/<name>.yaml）")
    parser.add_argument("--f1", action="store_true",
                        help="F1 校准模式（单基线配置）")
    parser.add_argument("--coarse", action="store_true",
                        help="粗扫模式（36 配置）")
    parser.add_argument("--fine", action="store_true",
                        help="细扫模式（100 配置）")
    parser.add_argument("--full", action="store_true",
                        help="完整扫描（80 配置）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只生成 YAML，不调用 Docker")
    parser.add_argument("--timeout", type=int, default=90,
                        help="每个算子的 Docker 超时（秒，默认 90）")
    parser.add_argument("--threads", type=int, default=1,
                        help="并发配置数（默认 1，串行）")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.f1:
        run_f1_calibration(
            chip=args.chip,
            model=args.model,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
    elif args.coarse:
        run_sweep(args.chip, args.model, "coarse",
                  args.timeout, args.dry_run, args.threads)
    elif args.fine:
        run_sweep(args.chip, args.model, "fine",
                  args.timeout, args.dry_run, args.threads)
    elif args.full:
        run_sweep(args.chip, args.model, "full",
                  args.timeout, args.dry_run, args.threads)
    else:
        parser.print_help()
        print("\n提示：使用 --f1 做 F1 校准，--coarse 做粗扫")


if __name__ == "__main__":
    main()
