#!/bin/bash
# Phase B 批量 msprof 采集（10 轮）
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dev_torch25
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/sim-experiment

BASE_FLAGS="--task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off"

run_msprof() {
    local model=$1; local batch=$2; local metric=$3; local loop=$4; local warmup=$5
    local outdir="msprof_${model}_b${batch}_${metric}"
    rm -rf "$outdir"
    mkdir -p "$outdir"
    chmod 750 "$outdir"
    echo "=== [$(date +%H:%M:%S)] Starting $outdir (loop=$loop) ===" >&2
    msprof --application="python3 -m ais_bench --model om/${model}_b${batch}.om --loop $loop --warmup_count $warmup" \
        --output="./$outdir" $BASE_FLAGS \
        --aic-metrics="$metric" --l2=on \
        2>&1 | tail -3 >&2
    echo "=== [$(date +%H:%M:%S)] Done $outdir ===" >&2
}

# BERT-base b1: 4 metrics
run_msprof bert_base 1 PipeUtilization 30 5
run_msprof bert_base 1 L2Cache 30 5
run_msprof bert_base 1 Memory 30 5
run_msprof bert_base 1 ArithmeticUtilization 30 5

# GPT-2-small b1: 2 metrics
run_msprof gpt2_small 1 PipeUtilization 20 5
run_msprof gpt2_small 1 L2Cache 20 5

# Qwen3 b1: 2 metrics
run_msprof qwen3_06b 1 PipeUtilization 8 3
run_msprof qwen3_06b 1 Memory 8 3

# Batch extension: BERT b16 + GPT-2 b16, 1 metric each (Memory)
run_msprof bert_base 16 Memory 15 5
run_msprof gpt2_small 16 Memory 10 3

echo "=== ALL DONE [$(date +%H:%M:%S)] ===" >&2
ls -la ~/sim-experiment/msprof_*_b*_* 2>/dev/null | grep '^d' | head -20
