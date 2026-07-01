#!/bin/bash
# Path B step 2/3: ATC sdpa-exported ONNX + msprof
cd ~/sim-experiment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dev_torch25
source /usr/local/Ascend/ascend-toolkit/set_env.sh

STEM="qwen3_06b_prefill_S4096_b1_sdpa"
ONNX="models/${STEM}.onnx"
OM="om/${STEM}.om"
OUTDIR="msprof_${STEM}_PipeUtilization"

echo "=== ATC start ($(date +%H:%M:%S)) ==="
[ ! -f "$ONNX" ] && { echo "✗ ONNX missing: $ONNX"; exit 1; }

# Capture FULL atc log to check for FlashAttention fusion messages
ATC_LOG="/tmp/atc_${STEM}.log"
atc --model="$ONNX" --framework=5 \
    --output="om/${STEM}" --soc_version=Ascend910B4 \
    --input_shape="input_ids:1,4096;attention_mask:1,4096" \
    --precision_mode=allow_fp32_to_fp16 \
    --log=info \
    2>&1 | tee "$ATC_LOG" | tail -8

if [ ! -f "$OM" ]; then
  echo "✗ ATC failed"
  exit 1
fi

echo ""
echo "=== ATC FlashAttention/PromptFlash mentions in log ==="
grep -iE "FlashAttention|PromptFlashAttention|attention.*fusion|fused.*attention" "$ATC_LOG" | head -15 || echo "(no FlashAttention mentions found)"

echo ""
echo "=== OM info ==="
ls -lh "$OM"

echo ""
echo "=== msprof PipeUtilization (LOOP=10) ==="
rm -rf "$OUTDIR"
msprof --application="python3 -m ais_bench --model $OM --loop 10 --warmup_count 2" \
       --output="./$OUTDIR" \
       --task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off \
       --aic-metrics=PipeUtilization --l2=on --analyze=on \
       2>&1 | tail -5

csv=$(find "$OUTDIR" -name 'op_summary*.csv' 2>/dev/null | head -1)
[ -n "$csv" ] && echo "✓ op_summary: $(wc -l < $csv) rows" || echo "✗ op_summary missing"

step=$(find "$OUTDIR" -name 'step_trace_*.csv' 2>/dev/null | head -1)
if [ -n "$step" ]; then
  echo "=== step_trace iter times (μs) ==="
  awk -F',' 'NR>1 && $0~/[0-9]/ {print $0}' "$step" | head -15
fi

echo "=== Done ($(date +%H:%M:%S)) ==="
