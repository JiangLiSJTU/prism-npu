#!/bin/bash
# ATC convert ModernBERT + SmolLM2 ONNX -> OM (Ascend 910B4)
# Usage: bash benchmark/convert_new_models_om.sh <onnx_dir> <om_dir>

ONNX_DIR="${1:-models}"
OM_DIR="${2:-om}"
SOC="Ascend910B4"

mkdir -p "$OM_DIR"

# ModernBERT prefill
for onnx_file in "$ONNX_DIR"/modernbert_base_prefill_S*_b*.onnx; do
    [ -f "$onnx_file" ] || continue
    base=$(basename "$onnx_file" .onnx)
    echo "=== $base ==="
    atc --model="$onnx_file" \
        --output="$OM_DIR/${base}" \
        --soc_version="$SOC" \
        --input_format=ND \
        --framework=5 \
        --log=error
done

# SmolLM2 decode
for onnx_file in "$ONNX_DIR"/smollm2_135m_decode_b*.onnx; do
    [ -f "$onnx_file" ] || continue
    base=$(basename "$onnx_file" .onnx)
    echo "=== $base ==="
    atc --model="$onnx_file" \
        --output="$OM_DIR/${base}" \
        --soc_version="$SOC" \
        --input_format=ND \
        --framework=5 \
        --log=error
done

echo "Done. OM files in $OM_DIR/"
ls -lh "$OM_DIR"/modernbert_* "$OM_DIR"/smollm2_* 2>/dev/null
