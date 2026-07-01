"""
昇腾 910B4 推理基准测试脚本
使用 ais_bench（官方工具）对 .om 模型做延迟测量

用法：
  # 先把 ONNX 转成 .om（见 convert_onnx_to_om.sh）
  python run_inference_ascend.py --model bert_base --batch 1 4 8 16
  python run_inference_ascend.py --model gpt2_small --batch 1 4 8 16
  python run_inference_ascend.py --model qwen3_06b --batch 1 4 8

依赖：
  - CANN 8.5.0（已装）
  - ais_bench（已装，随 CANN）
  - aclruntime（已装）

ONNX → .om 转换命令（在昇腾服务器上跑，或用 convert_onnx_to_om.sh）：
  # HF BERT 4层蒸馏版 batch=1
  atc --model=hf_bert_batch_1.onnx --framework=5 --output=hf_bert_b1 \
      --input_shape="input_ids:1,128;attention_mask:1,128" \
      --soc_version=Ascend910B4

  # ET-BERT batch=1（vocab=256，seq=128）
  atc --model=et_bert_batch_1.onnx --framework=5 --output=et_bert_b1 \
      --input_shape="input_ids:1,128;attention_mask:1,128" \
      --soc_version=Ascend910B4

  # MalConv2 batch=1（seq_len=1048576）
  atc --model=malconv2_batch_1.onnx --framework=5 --output=malconv2_b1 \
      --input_shape="file_bytes:1,1048576" \
      --soc_version=Ascend910B4

  # Kitsune batch=1（100维特征）
  atc --model=kitsune_batch_1.onnx --framework=5 --output=kitsune_b1 \
      --input_shape="features:1,100" \
      --soc_version=Ascend910B4

  # BERT-base batch=1（三输入，S=128）
  atc --model=bert_base_batch_1.onnx --framework=5 --output=bert_base_b1 \
      --input_shape="input_ids:1,128;attention_mask:1,128;token_type_ids:1,128" \
      --soc_version=Ascend910B4

  # GPT-2-small batch=1（S=512）
  atc --model=gpt2_small_batch_1.onnx --framework=5 --output=gpt2_small_b1 \
      --input_shape="input_ids:1,512" \
      --soc_version=Ascend910B4

  # Qwen3-0.6B batch=1（S=512；转换较慢，约 5-10 分钟）
  atc --model=qwen3_06b_batch_1.onnx --framework=5 --output=qwen3_06b_b1 \
      --input_shape="input_ids:1,512" \
      --soc_version=Ascend910B4

msprof 延迟分解（BERT-base 建议在测量后运行以验证 β_layer 线性假设）：
  source /usr/local/Ascend/cann-8.5.0/set_env.sh
  msprof --application="python3 -m ais_bench --model bert_base_b1.om --loop 50" \
         --output=./msprof_bert_base/ \
         --aic-metrics=PipeUtilization
  # 分析产物：msprof_bert_base/PROF_*/device_*/timeline/msprof.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path


def get_cann_version():
    """从环境变量或文件读取 CANN 版本"""
    toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
    if toolkit_home:
        # 从路径提取版本，如 /usr/local/Ascend/cann-8.5.0
        for part in toolkit_home.split("/"):
            if part.startswith("cann-"):
                return part.replace("cann-", "")
    return "8.5.0"


def run_ais_bench(om_path: str, batch: int, runs: int = 200, warmup: int = 20) -> dict:
    """
    用 ais_bench 测量 .om 模型的推理延迟，不写输出文件，直接解析 stdout。

    Returns:
        {"latency_ms": float, "latency_std_ms": float, "runs": int}
    """
    cmd = [
        "python3", "-m", "ais_bench",
        "--model", om_path,
        "--loop", str(runs),
        "--warmup_count", str(warmup),
    ]

    print(f"  运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # ais_bench 部分版本把结果打到 stderr，合并两路
    full_output = result.stdout + result.stderr

    if result.returncode != 0:
        # 若输出里有延迟数据则忽略非零返回码（部分版本会这样）
        if "NPU_MODEL_EXECUTE" not in full_output and "throughput" not in full_output.lower():
            print(f"  ais_bench 失败:\n{full_output[-800:]}")
            return None

    import re
    latency_ms = None
    latency_std_ms = None

    for line in full_output.split("\n"):
        # CANN 8.5 格式：
        # [INFO] NPU_compute_time (ms): min = X, max = X, mean = X, median = X, percentile(99%) = X
        if "NPU_compute_time" in line and "mean" in line:
            m = re.search(r"mean\s*=\s*([\d.]+)", line)
            if m:
                latency_ms = float(m.group(1))
            # std 用 max-min 范围近似（该格式无 std，用 0）
            latency_std_ms = 0.0
            break
        # 旧格式：NPU_MODEL_EXECUTE: throughput(fps): xxx, latency(ms): mean xxx, std xxx
        if "NPU_MODEL_EXECUTE" in line and "latency" in line:
            m_mean = re.search(r"mean\s+([\d.]+)", line)
            m_std  = re.search(r"std\s+([\d.]+)", line)
            if m_mean:
                latency_ms = float(m_mean.group(1))
            if m_std:
                latency_std_ms = float(m_std.group(1))
            break

    if latency_ms is None:
        print(f"  警告：无法解析延迟数据，原始输出：\n{full_output[-800:]}")
        return None

    return {
        "latency_ms": round(latency_ms, 4),
        "latency_std_ms": round(latency_std_ms or 0.0, 4),
        "runs": runs,
    }


def benchmark_model(model_name: str, om_files: dict, batches: list, runs: int = 200) -> dict:
    """
    对一个模型的多个 batch 做基准测试

    Args:
        model_name: 模型名称（用于输出 JSON）
        om_files: {batch: om_path} 映射
        batches: 要测试的 batch 列表
        runs: 每个 batch 的推理次数

    Returns:
        符合约定格式的 JSON dict
    """
    print(f"\n{'='*50}")
    print(f"模型: {model_name}")
    print(f"{'='*50}")

    results = []
    for batch in batches:
        if batch not in om_files:
            print(f"  batch={batch}: 跳过（无对应 .om 文件）")
            continue

        om_path = om_files[batch]
        if not Path(om_path).exists():
            print(f"  batch={batch}: 跳过（文件不存在: {om_path}）")
            continue

        print(f"\n  batch={batch}，跑 {runs} 次...")
        result = run_ais_bench(om_path, batch, runs=runs)

        if result:
            result["batch"] = batch
            results.append(result)
            print(f"  ✓ 延迟: {result['latency_ms']:.3f} ± {result['latency_std_ms']:.3f} ms")
        else:
            print(f"  ✗ batch={batch} 测量失败")

    return {
        "platform": "ascend_910b4",
        "model": model_name,
        "date": str(date.today()),
        "cann_version": get_cann_version(),
        "measurement_tool": "ais_bench",
        "precision": "FP16",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="昇腾 910B4 推理基准测试")
    parser.add_argument("--model", required=True,
                        choices=["hf_bert", "et_bert", "malconv2", "kitsune", "net_transformer",
                                 "bert_base", "gpt2_small", "qwen3_06b"],
                        help="要测试的模型")
    parser.add_argument("--om_dir", default=".",
                        help=".om 文件所在目录（默认当前目录）")
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 4, 8, 16],
                        help="要测试的 batch size 列表")
    parser.add_argument("--runs", type=int, default=200,
                        help="每个 batch 的推理次数（默认 200）")
    parser.add_argument("--output", default=None,
                        help="输出 JSON 文件路径（默认 results/<model>_910b4.json）")
    args = parser.parse_args()

    om_dir = Path(args.om_dir)

    # 各模型的 .om 文件名约定
    OM_FILE_PATTERNS = {
        # ── 原有模型 ──────────────────────────────────────────────
        "hf_bert":          {b: str(om_dir / f"hf_bert_b{b}.om")          for b in args.batch},
        "et_bert":          {b: str(om_dir / f"et_bert_b{b}.om")          for b in args.batch},
        "malconv2":         {b: str(om_dir / f"malconv2_b{b}.om")         for b in args.batch},
        "kitsune":          {b: str(om_dir / f"kitsune_b{b}.om")          for b in args.batch},
        "net_transformer":  {b: str(om_dir / f"net_transformer_b{b}.om")  for b in args.batch},
        # ── 校准扩展模型（Roofline 三-regime 校准）────────────────
        "bert_base":        {b: str(om_dir / f"bert_base_b{b}.om")        for b in args.batch},
        "gpt2_small":       {b: str(om_dir / f"gpt2_small_b{b}.om")       for b in args.batch},
        "qwen3_06b":        {b: str(om_dir / f"qwen3_06b_b{b}.om")        for b in args.batch},
    }

    om_files = OM_FILE_PATTERNS[args.model]
    data = benchmark_model(args.model, om_files, args.batch, runs=args.runs)

    # 输出路径
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("results") / f"{args.model}_910b4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {out_path}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
