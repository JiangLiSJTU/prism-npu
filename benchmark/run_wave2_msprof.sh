#!/bin/bash
# Wave 2 msprof phase only (ATC already done for hf_bert)
cd ~/sim-experiment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dev_torch25
source /usr/local/Ascend/ascend-toolkit/set_env.sh

BASE="--task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off"
LOOP=10
WARMUP=2

for OM_PATH in om/hf_bert_b1.om om/hf_bert_b4.om om/hf_bert_b8.om om/hf_bert_b16.om; do
  STEM=$(basename "$OM_PATH" .om)
  OUTDIR="msprof_${STEM}_PipeUtilization"
  rm -rf "$OUTDIR"
  echo "=== msprof: $STEM ($(date +%H:%M:%S)) ==="
  msprof --application="python3 -m ais_bench --model $OM_PATH --loop $LOOP --warmup_count $WARMUP" \
         --output="./$OUTDIR" $BASE \
         --aic-metrics=PipeUtilization --l2=on --analyze=on 2>&1 | tail -3 || echo "  ✗ failed"
done

echo "=== Wave 2 msprof done ($(date +%H:%M:%S)) ==="
ls -d msprof_hf_bert_b*_PipeUtilization 2>/dev/null
