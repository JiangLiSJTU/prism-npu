#!/usr/bin/env python3
"""
Phase I — Manual Mapping Runner（团队工具 v0.1）

封装 Docker timeloop-model 调用 + 解析 stats。配合 generate_manual_mapping.py 使用。

用法：
  # 一步流：生成 mapping + 跑 + 解析
  python run_manual_mapping.py \\
      --workload-name "qwen_emb_S4096_ffn_gate" \\
      --M 4096 --N 3072 --K 1024 \\
      --m-l2-spatial 4 --n-l2-spatial 6 \\
      --arch-yaml ../../arch/ascend_910b4_for_mapping.yaml

  # 仅跑已有 mapping：
  python run_manual_mapping.py \\
      --workload-name "..." \\
      --workload-yaml mapper/audit/<op>.yaml \\
      --mapping-yaml mapper/manual/<op>_24core.yaml \\
      --arch-yaml arch/ascend_910b4_for_mapping.yaml

注意（Phase L#2，2026-05-08）：
  推荐 `--arch-yaml arch/ascend_910b4_for_mapping.yaml`（L2 depth = 96 MB 真实值）。
  原 `arch/ascend_910b4.yaml` 是 locked 文件，L2 depth 写 1 MB 是 CACTI 限制妥协，
  manual mapping 会误以为 L2 只有 1MB 而过度切片。

输出：JSON 结果 + 控制台总结。
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# 同目录的 generator
HERE = Path(__file__).resolve().parent
from prism.mapper.generate import generate_mapping, ARCH_910B4

DOCKER_IMAGE = "timeloopaccelergy/accelergy-timeloop-infrastructure:latest-arm64"
COMPONENTS_DIR = "/usr/local/src/timeloop-python/tests/timeloop-accelergy-exercises/workspace/example_designs/example_designs/_components"
COMPONENTS = ["smartbuffer_SRAM.yaml", "intmac.yaml", "regfile.yaml"]


def make_workload_yaml(M: int, N: int, K: int, output_path: Path) -> None:
    """
    生成单 op GEMM workload yaml（Timeloop 格式：data_spaces 下划线，三层嵌套）。
    """
    from prism.sweep.timeloop_problem import convert_op_to_timeloop_problem
    import yaml

    op = {
        "name": "gemm",
        "shape": {
            "name": "GEMM",
            "dimensions": ["M", "N", "K"],
            "data-spaces": [
                {"name": "Weights", "projection": [[["K"]], [["N"]]]},
                {"name": "Inputs",  "projection": [[["M"]], [["K"]]]},
                {"name": "Outputs", "projection": [[["M"]], [["N"]]], "read-write": True},
            ],
        },
        "instance": {"M": M, "N": N, "K": K},
    }
    out = convert_op_to_timeloop_problem(op)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True)


def run_timeloop_model(arch_yaml: Path, workload_yaml: Path, mapping_yaml: Path,
                       run_dir: Path, timeout_sec: int = 120) -> dict:
    """
    通过 Docker 跑 timeloop-model，解析 stats。返回 {cycles, utilization, pj_per_compute, success}。
    """
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copy(arch_yaml, inputs / "arch.yaml")
    shutil.copy(workload_yaml, inputs / "model.yaml")
    shutil.copy(mapping_yaml, inputs / "manual_mapping.yaml")

    timeloop_cmd_inner = (
        "cd /home/workspace/run && timeloop-model "
        "/home/workspace/inputs/arch.yaml "
        "/home/workspace/inputs/model.yaml "
        "/home/workspace/inputs/manual_mapping.yaml "
        + " ".join(f"{COMPONENTS_DIR}/{c}" for c in COMPONENTS)
        + " 2>&1"
    )

    cmd = [
        "docker", "run", "--rm", "--entrypoint", "/bin/bash",
        "-v", f"{run_dir.resolve()}:/home/workspace/run",
        "-v", f"{inputs.resolve()}:/home/workspace/inputs:ro",
        DOCKER_IMAGE, "-c", timeloop_cmd_inner,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except FileNotFoundError as e:
        # docker binary missing
        return {
            "success": False, "error": f"docker not found: {e}",
            "cycles": None, "utilization": None, "pj_per_compute": None,
            "stderr_tail": "", "stdout_tail": "",
        }
    except subprocess.TimeoutExpired as e:
        return {
            "success": False, "error": f"timeloop-mapper timed out after {timeout_sec}s",
            "cycles": None, "utilization": None, "pj_per_compute": None,
            "stderr_tail": (e.stderr or b"").decode(errors="replace")[-300:],
            "stdout_tail": (e.stdout or b"").decode(errors="replace")[-300:],
        }
    stats_file = run_dir / "timeloop-model.stats.txt"

    result = {
        "success": False,
        "cycles": None,
        "utilization": None,
        "pj_per_compute": None,
        "stderr_tail": proc.stderr[-300:] if proc.stderr else "",
        "stdout_tail": proc.stdout[-300:] if proc.stdout else "",
    }

    if stats_file.exists():
        text = stats_file.read_text(encoding="utf-8")
        result["success"] = True
        m = re.search(r"^Cycles:\s*(\d+)", text, re.M)
        if m: result["cycles"] = int(m.group(1))
        m = re.search(r"^Utilization:\s*([\d.]+)%?", text, re.M)
        if m: result["utilization_pct"] = float(m.group(1))   # 已是百分比
        # fJ/Compute Total 行（在 stats 末尾）
        m = re.search(r"fJ/Compute.*?^\s*Total\s*=\s*([\d.]+)", text, re.M | re.S)
        if m: result["fj_per_compute"] = float(m.group(1))
        m = re.search(r"^Energy:\s*([\d.]+)\s*uJ", text, re.M)
        if m: result["energy_uj"] = float(m.group(1))

    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workload-name", required=True, help="标识符（用于 run dir 命名）")
    p.add_argument("--M", type=int, help="GEMM M（如指定，则自动 generate mapping）")
    p.add_argument("--N", type=int)
    p.add_argument("--K", type=int)
    p.add_argument("--m-l2-spatial", type=int, default=4)
    p.add_argument("--n-l2-spatial", type=int, default=6)
    p.add_argument("--m-cube-spatial", type=int, default=16)
    p.add_argument("--n-cube-spatial", type=int, default=16)
    p.add_argument("--workload-yaml", help="已有 workload yaml（跳过 auto-gen）")
    p.add_argument("--mapping-yaml", help="已有 manual mapping yaml（跳过 auto-gen）")
    p.add_argument("--arch-yaml", required=True, help="Timeloop arch yaml（含 components）")
    p.add_argument("--run-dir", default=None, help="运行目录（默认 timeloop_results/manual_mapping/<name>）")
    p.add_argument("--cube-k-correction", type=int, default=16,
                   help="Timeloop K-temporal vs Cube K-spatial 修正因子（默认 16，cycles / 16 = 真实 wall-clock cycles）")
    p.add_argument("--clock-mhz", type=int, default=1000, help="时钟频率（用于 wall-clock μs 计算，默认 1 GHz）")
    p.add_argument("--output-json", default=None, help="结果 JSON 输出路径")
    args = p.parse_args()

    repo_sim = HERE.parent.parent  # sim-experiment/
    run_dir = Path(args.run_dir) if args.run_dir else (
        repo_sim / "timeloop_results" / "manual_mapping" / args.workload_name
    )
    run_dir = run_dir.resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    # 1. 准备 workload yaml
    if args.workload_yaml:
        workload_yaml = Path(args.workload_yaml).resolve()
    else:
        if not (args.M and args.N and args.K):
            p.error("需要 --workload-yaml 或 (--M --N --K)")
        workload_yaml = run_dir / "workload.yaml"
        make_workload_yaml(args.M, args.N, args.K, workload_yaml)

    # 2. 准备 mapping yaml
    if args.mapping_yaml:
        mapping_yaml = Path(args.mapping_yaml).resolve()
    else:
        if not (args.M and args.N and args.K):
            p.error("auto-gen mapping 需要 --M --N --K")
        mapping_yaml = run_dir / "mapping.yaml"
        yaml_str, info = generate_mapping(
            M=args.M, N=args.N, K=args.K,
            m_l2_spatial=args.m_l2_spatial, n_l2_spatial=args.n_l2_spatial,
            m_cube_spatial=args.m_cube_spatial, n_cube_spatial=args.n_cube_spatial,
        )
        mapping_yaml.write_text(yaml_str, encoding="utf-8")

    # 3. 跑 Docker timeloop-model
    arch_yaml = Path(args.arch_yaml).resolve()
    print(f"Running timeloop-model on {args.workload_name} ...")
    result = run_timeloop_model(arch_yaml, workload_yaml, mapping_yaml, run_dir)

    # 4. 后处理：应用 Cube K 修正因子
    if result["success"] and result["cycles"]:
        cycles_raw = result["cycles"]
        cycles_corrected = cycles_raw // args.cube_k_correction
        wall_clock_us_corrected = cycles_corrected / (args.clock_mhz * 1000) * 1e6  # cycles / clock_hz * 1e6 = us... wait
        # cycles / (MHz × 1e6) × 1e6 us = cycles / MHz
        wall_clock_us_corrected = cycles_corrected / args.clock_mhz   # cycles / MHz = us
        wall_clock_us_raw       = cycles_raw       / args.clock_mhz

        result.update({
            "cycles_raw": cycles_raw,
            "cycles_corrected": cycles_corrected,
            "wall_clock_us_raw": wall_clock_us_raw,
            "wall_clock_us_corrected": wall_clock_us_corrected,
            "cube_k_correction_factor": args.cube_k_correction,
            "clock_mhz": args.clock_mhz,
        })

        print(f"\n=== Result ({args.workload_name}) ===")
        print(f"  cycles (raw):        {cycles_raw:>15,d}")
        print(f"  utilization:         {result.get('utilization_pct', 0):>14.2f}%")
        print(f"  fJ/Compute:          {result.get('fj_per_compute', 0):>14.2f}")
        print(f"  energy:              {result.get('energy_uj', 0):>14.2f} uJ")
        print(f"  cycles (corrected):  {cycles_corrected:>15,d}    (raw / {args.cube_k_correction})")
        print(f"  wall-clock raw:      {wall_clock_us_raw:>14.2f} μs @{args.clock_mhz} MHz")
        print(f"  wall-clock corrected:{wall_clock_us_corrected:>14.2f} μs @{args.clock_mhz} MHz")
    else:
        print(f"\nERROR: timeloop-model failed")
        print(f"stderr: {result['stderr_tail']}")
        print(f"stdout: {result['stdout_tail']}")
        return 1

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nResult JSON: {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
