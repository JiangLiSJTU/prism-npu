#!/bin/bash
# Wave 1: bert_base + gpt2_small PipeUtilization for b=4/8/16
# Existing OM files; just need PipeUtilization metric to fill the gap.
set -e
cd ~/sim-experiment

# Setup environment quietly
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dev_torch25
source /usr/local/Ascend/ascend-toolkit/set_env.sh

BASE="--task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off"
LOOP=10
WARMUP=2

# 6 new PipeUtilization configs
for MODEL in bert_base gpt2_small; do
  for B in 4 8 16; do
    OUTDIR="msprof_${MODEL}_b${B}_PipeUtilization"
    if [ -d "$OUTDIR/PROF_"* ] 2>/dev/null; then
      echo "[skip] $OUTDIR exists"
      continue
    fi
    rm -rf "$OUTDIR"
    OM="om/${MODEL}_b${B}.om"
    if [ ! -f "$OM" ]; then
      echo "[miss] $OM"
      continue
    fi
    echo "=== msprof: $MODEL b=$B PipeUtilization ($(date +%H:%M:%S)) ==="
    msprof --application="python3 -m ais_bench --model $OM --loop $LOOP --warmup_count $WARMUP" \
           --output="./$OUTDIR" $BASE \
           --aic-metrics=PipeUtilization --l2=on --analyze=on 2>&1 | tail -3 || echo "  ✗ failed"
  done
done

echo ""
echo "=== Wave 1 done ($(date +%H:%M:%S)) ==="
ls -d msprof_{bert_base,gpt2_small}_b{4,8,16}_PipeUtilization 2>/dev/null
