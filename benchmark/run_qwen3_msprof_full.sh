#!/bin/bash
# Qwen3-0.6B (S=512 prefill) × b=1/4/8 × 4 metrics msprof 全采集
# 已存在则跳过（按目录名）
set -e
cd ~/sim-experiment
SETUP="source ~/miniconda3/etc/profile.d/conda.sh && conda activate dev_torch25 && source /usr/local/Ascend/ascend-toolkit/set_env.sh"
eval "$SETUP"

BASE="--task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off"
LOOP=50
WARMUP=10

declare -A METRIC_FLAGS=(
  ["PipeUtilization"]="--aic-metrics=PipeUtilization --l2=on"
  ["ArithmeticUtilization"]="--aic-metrics=ArithmeticUtilization"
  ["L2Cache"]="--aic-metrics=L2Cache --l2=on"
  ["Memory"]="--aic-metrics=Memory --sys-hardware-mem=on"
)

for B in 1 4 8; do
  for M in PipeUtilization ArithmeticUtilization L2Cache Memory; do
    OUTDIR="msprof_qwen3_06b_b${B}_${M}"
    if [ -d "$OUTDIR/PROF_"* ] 2>/dev/null; then
      echo "[skip] $OUTDIR 已存在"
      continue
    fi
    rm -rf "$OUTDIR"
    OM="om/qwen3_06b_b${B}.om"
    if [ ! -f "$OM" ]; then
      echo "[miss] $OM 不存在，跳过"
      continue
    fi

    echo "=== msprof: b=${B} metric=${M} ==="
    msprof --application="python3 -m ais_bench --model $OM --loop $LOOP --warmup_count $WARMUP" \
           --output="./$OUTDIR" $BASE \
           ${METRIC_FLAGS[$M]} --analyze=on 2>&1 | tail -3 || echo "  ✗ failed"
  done
done

echo ""
echo "=== 采集完成，结果目录 ==="
ls -d msprof_qwen3_06b_b*_* 2>/dev/null
