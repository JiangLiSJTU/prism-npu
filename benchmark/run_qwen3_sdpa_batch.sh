#!/bin/bash
# Phase 1: SDPA re-export + ATC + msprof for 5 remaining Qwen3 configs
cd ~/sim-experiment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dev_torch25
source /usr/local/Ascend/ascend-toolkit/set_env.sh

LOOP=10
WARMUP=2
SOC="Ascend910B4"

# (S, batch) pairs
configs=(
  "256 1"
  "256 4"
  "256 8"
  "512 4"
  "512 8"
)

echo "=== Phase 1 SDPA batch start ($(date +%H:%M:%S)) ==="

for cfg in "${configs[@]}"; do
  S=$(echo $cfg | awk '{print $1}')
  B=$(echo $cfg | awk '{print $2}')
  STEM="qwen3_06b_prefill_S${S}_b${B}_sdpa"
  ONNX="models/${STEM}.onnx"
  OM="om/${STEM}.om"
  OUTDIR="msprof_${STEM}_PipeUtilization"

  echo ""
  echo "─── S=$S batch=$B ──── $(date +%H:%M:%S) ───"

  # 1. Export
  if [ ! -f "$ONNX" ]; then
    echo "  [export]"
    python3 benchmark/export_qwen3_prefill_sdpa.py --S $S --batch $B 2>&1 | tail -3
  fi
  [ ! -f "$ONNX" ] && { echo "  ✗ export failed"; continue; }
  echo "  ONNX: $(ls -lh $ONNX | awk '{print $5}')"

  # 2. ATC
  if [ ! -f "$OM" ]; then
    echo "  [atc]"
    atc --model="$ONNX" --framework=5 \
        --output="om/${STEM}" --soc_version=$SOC \
        --input_shape="input_ids:${B},${S};attention_mask:${B},${S}" \
        --precision_mode=allow_fp32_to_fp16 2>&1 | tail -3
  fi
  [ ! -f "$OM" ] && { echo "  ✗ atc failed"; continue; }
  echo "  OM:  $(ls -lh $OM | awk '{print $5}')"

  # 3. msprof
  rm -rf "$OUTDIR"
  echo "  [msprof]"
  msprof --application="python3 -m ais_bench --model $OM --loop $LOOP --warmup_count $WARMUP" \
         --output="./$OUTDIR" \
         --task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off \
         --aic-metrics=PipeUtilization --l2=on --analyze=on 2>&1 | tail -2

  csv=$(find "$OUTDIR" -name 'op_summary*.csv' 2>/dev/null | head -1)
  step=$(find "$OUTDIR" -name 'step_trace_*.csv' 2>/dev/null | head -1)
  if [ -n "$csv" ] && [ -n "$step" ]; then
    iter_us=$(awk -F',' 'NR==2 && $5 ~ /[0-9]/ {print int($5)}' "$step")
    echo "  ✓ op_summary $(wc -l < $csv) rows; iter≈${iter_us}μs"
  else
    echo "  ✗ msprof missing csv"
  fi
done

echo ""
echo "=== Phase 1 SDPA batch done ($(date +%H:%M:%S)) ==="
ls -la om/qwen3*sdpa*.om 2>/dev/null
