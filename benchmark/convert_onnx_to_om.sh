#!/bin/bash
# convert_onnx_to_om.sh
# 把本地 ONNX 文件批量转换为昇腾 .om 格式
#
# 用法：
#   bash convert_onnx_to_om.sh <onnx_dir> <output_dir>
#
# 示例：
#   bash convert_onnx_to_om.sh ~/onnx_models ~/om_models
#
# 前提：
#   - CANN 8.5.0 已激活（atc 命令可用）
#   - ONNX 文件已上传到 onnx_dir

set -e

ONNX_DIR="${1:-.}"
OUTPUT_DIR="${2:-.}"
SOC="Ascend910B4"

mkdir -p "$OUTPUT_DIR"

echo "ONNX 目录: $ONNX_DIR"
echo "输出目录:  $OUTPUT_DIR"
echo "芯片型号:  $SOC"
echo ""

# ── HF BERT 4层蒸馏版 ──────────────────────────────────────────
for B in 1 4 8; do
    ONNX="$ONNX_DIR/hf_bert_batch_${B}.onnx"
    OUT="$OUTPUT_DIR/hf_bert_b${B}"
    if [ -f "$ONNX" ]; then
        echo "[HF BERT batch=$B] 转换中..."
        atc --model="$ONNX" \
            --framework=5 \
            --output="$OUT" \
            --input_shape="input_ids:${B},128;attention_mask:${B},128" \
            --soc_version="$SOC" \
            --log=error
        echo "  ✓ $OUT.om"
    else
        echo "  跳过 HF BERT batch=$B（文件不存在: $ONNX）"
    fi
done

# ── ET-BERT（vocab=256，BERT-base 架构）────────────────────────
for B in 1 4 8 16; do
    ONNX="$ONNX_DIR/et_bert_batch_1.onnx"   # 只有 batch=1 的 ONNX，用动态 batch
    OUT="$OUTPUT_DIR/et_bert_b${B}"
    if [ -f "$ONNX" ]; then
        echo "[ET-BERT batch=$B] 转换中..."
        atc --model="$ONNX" \
            --framework=5 \
            --output="$OUT" \
            --input_shape="input_ids:${B},128;attention_mask:${B},128" \
            --soc_version="$SOC" \
            --log=error
        echo "  ✓ $OUT.om"
    else
        echo "  跳过 ET-BERT batch=$B（文件不存在: $ONNX）"
    fi
done

# ── MalConv2（1D CNN，seq_len=1MB）────────────────────────────
ONNX="$ONNX_DIR/malconv2_batch_1.onnx"
OUT="$OUTPUT_DIR/malconv2_b1"
if [ -f "$ONNX" ]; then
    echo "[MalConv2 batch=1] 转换中（输入 1MB，可能较慢）..."
    atc --model="$ONNX" \
        --framework=5 \
        --output="$OUT" \
        --input_shape="file_bytes:1,1048576" \
        --soc_version="$SOC" \
        --log=error
    echo "  ✓ $OUT.om"
else
    echo "  跳过 MalConv2（文件不存在: $ONNX）"
fi

# ── Net-Transformer（GQA，输入 [B, 256, 384]）─────────────────
# 注意：先在本地跑 export_net_transformer.py 生成 ONNX，再传到服务器转换
for B in 1 4 8 16; do
    ONNX="$ONNX_DIR/net_transformer_batch_${B}.onnx"
    OUT="$OUTPUT_DIR/net_transformer_b${B}"
    if [ -f "$ONNX" ]; then
        echo "[Net-Transformer batch=$B] 转换中..."
        atc --model="$ONNX" \
            --framework=5 \
            --output="$OUT" \
            --input_shape="input_features:${B},256,384" \
            --soc_version="$SOC" \
            --log=error
        echo "  ✓ $OUT.om"
    else
        echo "  跳过 Net-Transformer batch=$B（文件不存在: $ONNX）"
    fi
done

# ── Kitsune（集成自编码器，100维特征）─────────────────────────
for B in 1 64; do
    ONNX="$ONNX_DIR/kitsune_batch_1.onnx"
    OUT="$OUTPUT_DIR/kitsune_b${B}"
    if [ -f "$ONNX" ]; then
        echo "[Kitsune batch=$B] 转换中..."
        atc --model="$ONNX" \
            --framework=5 \
            --output="$OUT" \
            --input_shape="features:${B},100" \
            --soc_version="$SOC" \
            --log=error
        echo "  ✓ $OUT.om"
    else
        echo "  跳过 Kitsune batch=$B（文件不存在: $ONNX）"
    fi
done

# ── BERT-base（google-bert/bert-base-uncased，标准 12 层）──────
# 来源：export_bert_gpt2_qwen.py；三输入：input_ids, attention_mask, token_type_ids
for B in 1 4 8 16; do
    ONNX="$ONNX_DIR/bert_base_batch_${B}.onnx"
    OUT="$OUTPUT_DIR/bert_base_b${B}"
    if [ -f "$ONNX" ]; then
        echo "[BERT-base batch=$B] 转换中..."
        atc --model="$ONNX" \
            --framework=5 \
            --output="$OUT" \
            --input_shape="input_ids:${B},128;attention_mask:${B},128;token_type_ids:${B},128" \
            --soc_version="$SOC" \
            --log=error
        echo "  ✓ $OUT.om"
    else
        echo "  跳过 BERT-base batch=$B（文件不存在: $ONNX）"
    fi
done

# ── GPT-2-small（openai-community/gpt2，12 层，S=512）──────────
# use_cache=False，单输入 input_ids；与 BERT-base 同 L=12 对照（causal vs bidirectional）
for B in 1 4 8 16; do
    ONNX="$ONNX_DIR/gpt2_small_batch_${B}.onnx"
    OUT="$OUTPUT_DIR/gpt2_small_b${B}"
    if [ -f "$ONNX" ]; then
        echo "[GPT-2-small batch=$B] 转换中..."
        atc --model="$ONNX" \
            --framework=5 \
            --output="$OUT" \
            --input_shape="input_ids:${B},512" \
            --soc_version="$SOC" \
            --log=error
        echo "  ✓ $OUT.om"
    else
        echo "  跳过 GPT-2-small batch=$B（文件不存在: $ONNX）"
    fi
done

# ── Qwen3-0.6B（L=28，GQA 16Q/8KV，SwiGLU，S=512）────────────
# use_cache=False；权重约 1.2 GB，内存受限 regime 校准点
# batch=8 时约需 ~10 GB DRAM，确认服务器内存充足再跑
for B in 1 4 8; do
    ONNX="$ONNX_DIR/qwen3_06b_batch_${B}.onnx"
    OUT="$OUTPUT_DIR/qwen3_06b_b${B}"
    if [ -f "$ONNX" ]; then
        echo "[Qwen3-0.6B batch=$B] 转换中（ONNX 文件较大，可能需要 5-10 分钟）..."
        atc --model="$ONNX" \
            --framework=5 \
            --output="$OUT" \
            --input_shape="input_ids:${B},512" \
            --soc_version="$SOC" \
            --log=error
        echo "  ✓ $OUT.om"
    else
        echo "  跳过 Qwen3-0.6B batch=$B（文件不存在: $ONNX）"
    fi
done

echo ""
echo "转换完成。生成的 .om 文件："
ls -lh "$OUTPUT_DIR"/*.om 2>/dev/null || echo "（无 .om 文件生成）"
