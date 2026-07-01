#!/bin/bash
# Phase N — PipeUtilization 补采（Qwen3 prefill 系列 8 配置）
#
# 用途：补齐 Phase M+ 缺失的 PipeUtilization 数据，让 Phase N
#       β_layer 分解审计在 prefill / 长上下文 workload 上有数据点。
#
# 已有 PipeUtil（无需重采）：
#   bert_base b=1, gpt2_small b=1, qwen3_06b decode b=1/4/8
#
# 本脚本采集（8 配置 × ~2-3 min/配置 = ~25 min NPU）：
#   qwen3_prefill_S256_b1   PipeUtilization  loop=10 warmup=2
#   qwen3_prefill_S256_b4   PipeUtilization  loop=10 warmup=2
#   qwen3_prefill_S512_b1   PipeUtilization  loop=10 warmup=2
#   qwen3_prefill_S512_b4   PipeUtilization  loop=10 warmup=2
#   qwen3_prefill_S4096_b1  PipeUtilization  loop=5  warmup=1
#   qwen3_prefill_S4096_b4  PipeUtilization  loop=5  warmup=1
#   qwen3_prefill_S8192_b1  PipeUtilization  loop=2  warmup=1   (may fail at analyze)
#   qwen3_prefill_S8192_b4  PipeUtilization  loop=1  warmup=0   (likely fails)
#
# 已知崩溃模式（参考 Phase M+ 经验）：
#   - S=4096 B=8 / S=8192 B=1 在 ArithUtil 上即便 loop=2 也崩
#   - PipeUtilization 数据量更大（更多列），对 analyze 阶段更敏感
#   - 若大 S/B 失败，接受数据缺失，仅用小 S/B 做 audit
#
# 前置条件：
#   1) ~/sim-experiment/om/qwen3_prefill_S{S}_b{B}.om 8 个 OM 文件齐全
#   2) ais_bench / msprof / CANN 8.5 环境就绪
#
# 运行：
#   ssh user@npu-server
#   cd ~/sim-experiment
#   bash benchmark/run_pipeutil_supplement.sh 2>&1 | tee pipeutil_supplement.log
#
# 取回：
#   rsync -avz user@npu-server:~/sim-experiment/msprof_qwen3_prefill_S*_b*_PipeUtilization/ \
#         msprof_data/

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dev_torch25
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/sim-experiment

BASE_FLAGS="--task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off"

run_msprof() {
    local s=$1; local batch=$2; local loop=$3; local warmup=$4
    local model="qwen3_prefill_S${s}"
    local outdir="msprof_${model}_b${batch}_PipeUtilization"
    local omfile="om/${model}_b${batch}.om"

    if [[ ! -f "$omfile" ]]; then
        echo "=== [SKIP] $omfile not found ===" >&2
        return
    fi

    rm -rf "$outdir"
    mkdir -p "$outdir"
    chmod 750 "$outdir"
    echo "=== [$(date +%H:%M:%S)] $outdir (loop=$loop warmup=$warmup) ===" >&2

    # 用 || true 容错：大 S/B 在 analyze 阶段崩溃不阻断后续配置
    msprof --application="python3 -m ais_bench --model $omfile --loop $loop --warmup_count $warmup" \
        --output="./$outdir" $BASE_FLAGS \
        --aic-metrics="PipeUtilization" --l2=on \
        2>&1 | tail -3 >&2 || echo "=== [WARN] $outdir failed ===" >&2

    echo "=== [$(date +%H:%M:%S)] Done $outdir ===" >&2
}

# 短/中 S：保守 loop=10 warmup=2
run_msprof 256  1 10 2
run_msprof 256  4 10 2
run_msprof 512  1 10 2
run_msprof 512  4 10 2

# 长 S：减小到 loop=5 warmup=1
run_msprof 4096 1 5 1
run_msprof 4096 4 5 1

# 极长 S：可能崩溃，loop=2/1
run_msprof 8192 1 2 1
run_msprof 8192 4 1 0

echo "=== ALL DONE [$(date +%H:%M:%S)] ===" >&2
ls -la ~/sim-experiment/msprof_qwen3_prefill_S*_b*_PipeUtilization 2>/dev/null | grep '^d' | head -20
echo ""
echo "=== 检查每个目录是否有 op_summary CSV ==="
for d in msprof_qwen3_prefill_S*_b*_PipeUtilization; do
    csv_count=$(find "$d" -name "op_summary*.csv" 2>/dev/null | wc -l)
    echo "  $d : op_summary CSV count = $csv_count"
done
