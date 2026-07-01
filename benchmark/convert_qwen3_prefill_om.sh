#!/bin/bash
# 把 Qwen3-0.6B prefill ONNX 文件批量转 OM
# 用法：bash convert_qwen3_prefill_om.sh <onnx_dir> <om_dir> "<S list>" "<B list>"
# 示例：bash convert_qwen3_prefill_om.sh ~/sim-experiment/models ~/sim-experiment/om "256 512 4096" "1"

set -e
ONNX_DIR="${1:-$HOME/sim-experiment/models}"
OM_DIR="${2:-$HOME/sim-experiment/om}"
S_LIST="${3:-256 512 4096 8192}"
B_LIST="${4:-1}"
SOC="Ascend910B4"

mkdir -p "$OM_DIR"
echo "ONNX dir: $ONNX_DIR"
echo "OM dir:   $OM_DIR"
echo "S = $S_LIST,  B = $B_LIST"

for S in $S_LIST; do
  for B in $B_LIST; do
    ONNX="$ONNX_DIR/qwen3_06b_prefill_S${S}_b${B}.onnx"
    OUT_BASE="$OM_DIR/qwen3_06b_prefill_S${S}_b${B}"

    if [ ! -f "$ONNX" ]; then
      echo "  跳过 S=$S B=$B（ONNX 不存在: $ONNX）"
      continue
    fi
    if [ -f "$OUT_BASE.om" ]; then
      echo "  已存在 $OUT_BASE.om，跳过"
      continue
    fi

    echo "[Qwen3 prefill S=$S B=$B] ATC 转换中（5-10 分钟）..."
    atc --model="$ONNX" \
        --framework=5 \
        --output="$OUT_BASE" \
        --input_shape="input_ids:${B},${S}" \
        --soc_version="$SOC" \
        --log=error
    echo "  ✓ $OUT_BASE.om"
  done
done

echo ""
echo "=== 生成的 OM ==="
ls -lh "$OM_DIR"/qwen3_06b_prefill_*.om 2>/dev/null
