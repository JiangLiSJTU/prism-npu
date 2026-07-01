#!/bin/bash
# Wave 2: ATC + msprof PipeUtil for net_transformer / hf_bert / malconv2 / kitsune
set -e
cd ~/sim-experiment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dev_torch25
source /usr/local/Ascend/ascend-toolkit/set_env.sh

SOC="Ascend910B4"
BASE="--task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off"
LOOP=10
WARMUP=2

mkdir -p om

# ── ATC conversions ───────────────────────────────────────────────
echo "=== Wave 2 ATC phase ($(date +%H:%M:%S)) ==="

# net_transformer (4 batches; assume input_ids with [B, S=256] from name)
for B in 1 4 8 16; do
  ONNX="models/net_transformer_batch_${B}.onnx"
  OM_OUT="om/net_transformer_b${B}"
  if [ -f "${OM_OUT}.om" ]; then echo "[skip ATC] net_transformer_b${B}"; continue; fi
  if [ ! -f "$ONNX" ]; then echo "[miss ONNX] $ONNX"; continue; fi
  echo "  ATC net_transformer b=$B"
  # net_transformer expected shape: usually [B, 256] for S=256, single int64 input
  # Use auto shape detection if --input_shape not specifiable
  atc --model="$ONNX" --framework=5 --output="$OM_OUT" --soc_version="$SOC" \
      --precision_mode=allow_fp32_to_fp16 2>&1 | tail -2 || echo "  ✗ ATC failed"
done

# hf_bert (4 batches, S=128 standard)
for B in 1 4 8 16; do
  ONNX="models/hf_bert_batch_${B}.onnx"
  OM_OUT="om/hf_bert_b${B}"
  if [ -f "${OM_OUT}.om" ]; then echo "[skip ATC] hf_bert_b${B}"; continue; fi
  if [ ! -f "$ONNX" ]; then echo "[miss ONNX] $ONNX"; continue; fi
  echo "  ATC hf_bert b=$B"
  atc --model="$ONNX" --framework=5 --output="$OM_OUT" --soc_version="$SOC" \
      --precision_mode=allow_fp32_to_fp16 2>&1 | tail -2 || echo "  ✗ ATC failed"
done

# malconv2, kitsune (single batch each, likely b=1)
for MODEL in malconv2 kitsune; do
  ONNX="models/${MODEL}_batch_1.onnx"
  OM_OUT="om/${MODEL}_b1"
  if [ -f "${OM_OUT}.om" ]; then echo "[skip ATC] ${MODEL}_b1"; continue; fi
  if [ ! -f "$ONNX" ]; then echo "[miss ONNX] $ONNX"; continue; fi
  echo "  ATC $MODEL b=1"
  atc --model="$ONNX" --framework=5 --output="$OM_OUT" --soc_version="$SOC" \
      --precision_mode=allow_fp32_to_fp16 2>&1 | tail -2 || echo "  ✗ ATC failed"
done

echo "=== ATC done ($(date +%H:%M:%S)) ==="
ls -lh om/{net_transformer,hf_bert,malconv2,kitsune}_b*.om 2>/dev/null

# ── msprof PipeUtilization ─────────────────────────────────────────
echo ""
echo "=== Wave 2 msprof PipeUtilization phase ($(date +%H:%M:%S)) ==="
for OM_PATH in om/net_transformer_b1.om om/net_transformer_b4.om om/net_transformer_b8.om om/net_transformer_b16.om \
               om/hf_bert_b1.om om/hf_bert_b4.om om/hf_bert_b8.om om/hf_bert_b16.om \
               om/malconv2_b1.om om/kitsune_b1.om; do
  if [ ! -f "$OM_PATH" ]; then echo "[miss OM] $OM_PATH"; continue; fi
  STEM=$(basename "$OM_PATH" .om)
  OUTDIR="msprof_${STEM}_PipeUtilization"
  if [ -d "$OUTDIR/PROF_"* ] 2>/dev/null; then echo "[skip msprof] $OUTDIR"; continue; fi
  rm -rf "$OUTDIR"
  echo "=== msprof: $STEM PipeUtilization ($(date +%H:%M:%S)) ==="
  msprof --application="python3 -m ais_bench --model $OM_PATH --loop $LOOP --warmup_count $WARMUP" \
         --output="./$OUTDIR" $BASE \
         --aic-metrics=PipeUtilization --l2=on --analyze=on 2>&1 | tail -3 || echo "  ✗ msprof failed"
done

echo ""
echo "=== Wave 2 done ($(date +%H:%M:%S)) ==="
ls -d msprof_{net_transformer,hf_bert,malconv2,kitsune}_b*_PipeUtilization 2>/dev/null
