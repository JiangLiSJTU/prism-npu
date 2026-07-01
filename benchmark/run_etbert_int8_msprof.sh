#!/bin/bash
# ET-BERT — 补齐仓库三项空白之 ③INT8 实测(当前 aic_mac_int8_ratio 全为 0)
#
# 背景：仓库所有实测 config 都是 FP16，cube_util_int8_pct 恒为 0 —— 从未在 910B 上
#       跑过 INT8 负载。而 INT8 是 910B 原生支持(560 TOPS)、且是验证 FP8/FP4 外推
#       方向的**唯一可在现有硅上实测**的精度档。本脚本做 FP16↔INT8 对照实测：
#       量化(AMCT PTQ) → ATC 转 OM → msprof 三组指标 → 解析对比。
#
# 验证目标(对应 Q2/Q3 的 PRISM 外推)：
#   • cube_util_int8_ratio > 0(FP16 时为 0)          —— 量化确实落到 INT8 cube
#   • aic_pipes.mte2(HBM 权重流量) INT8 ≈ FP16 的 1/2  —— 位宽减半→访存减半
#   • 同 wall-clock 下 cube-busy% 变化                —— 精度↓ 对 MFU 的真实影响
#   • L2 命中率 / DDR 利用率 随精度变化                —— 配合 run_etbert_llc_ddr
#
# 前置条件(910B 真机)：
#   1) benchmark/export_models_onnx.py 已生成 onnx/et_bert_batch_1.onnx
#   2) AMCT(Ascend Model Compression Toolkit, amct_onnx) 已安装(随 CANN)
#   3) CANN 8.5 + ais_bench + msprof 就绪；SOC=Ascend910B4
#   4) 校准数据 calib_data/(几百条 128-len 字节级 token 序列;可用真实流量样本)
#
# 运行：
#   ssh user@npu-server && cd ~/sim-experiment
#   bash benchmark/run_etbert_int8_msprof.sh 2>&1 | tee etbert_int8.log
#
# 取回：
#   rsync -avz user@npu-server:~/sim-experiment/msprof_et_bert_{fp16,int8}_b*_*/ msprof_data/

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dev_torch25
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/sim-experiment

SOC="Ascend910B4"
ONNX="onnx/et_bert_batch_1.onnx"          # 动态 batch；input_shape 在 ATC 指定
SHAPE_TMPL="input_ids:%d,128;attention_mask:%d,128"
BASE_FLAGS="--task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off"

# ── 0) 确保 FP16 与 INT8 两套 OM 就绪 ─────────────────────────────────────
mkdir -p om amct_out

build_fp16_om() {
    local B=$1; local out="om/et_bert_fp16_b${B}"
    [[ -f "${out}.om" ]] && { echo "[FP16 b$B] OM 已存在,跳过"; return; }
    echo "[FP16 b$B] ATC 转换..."
    atc --model="$ONNX" --framework=5 --output="$out" --soc_version="$SOC" \
        --input_shape="$(printf "$SHAPE_TMPL" "$B" "$B")"
}

# AMCT PTQ：用校准集统计激活/权重分布 → 生成 INT8 量化 ONNX(deploy 模型) → ATC
build_int8_om() {
    local B=$1
    local q="amct_out/et_bert_int8_b${B}"; local out="om/et_bert_int8_b${B}"
    [[ -f "${out}.om" ]] && { echo "[INT8 b$B] OM 已存在,跳过"; return; }
    if [[ ! -f "${q}_deploy_model.onnx" ]]; then
        echo "[INT8 b$B] AMCT PTQ 校准..."
        # amct_onnx 版本接口略有差异,以下为 CANN 8.x 典型 PTQ 调用;
        # 若接口不符,参见 `amct_onnx calibration --help`。
        amct_onnx calibration \
            --model="$ONNX" \
            --save_path="$q" \
            --input_shape="$(printf "$SHAPE_TMPL" "$B" "$B")" \
            --data_dir="calib_data" \
            --data_types="int8"
    fi
    echo "[INT8 b$B] ATC 转换(量化后 deploy onnx)..."
    atc --model="${q}_deploy_model.onnx" --framework=5 --output="$out" --soc_version="$SOC" \
        --input_shape="$(printf "$SHAPE_TMPL" "$B" "$B")"
}

# ── 1) msprof 采集(三组指标) ─────────────────────────────────────────────
run_msprof() {
    local prec=$1; local B=$2; local metric=$3; local loop=$4; local warmup=$5
    local omfile="om/et_bert_${prec}_b${B}.om"
    local outdir="msprof_et_bert_${prec}_b${B}_${metric}"
    [[ -f "$omfile" ]] || { echo "=== [SKIP] $omfile ===" >&2; return; }
    rm -rf "$outdir"; mkdir -p "$outdir"; chmod 750 "$outdir"
    echo "=== [$(date +%H:%M:%S)] $outdir (loop=$loop) ===" >&2
    msprof --application="python3 -m ais_bench --model $omfile --loop $loop --warmup_count $warmup" \
        --output="./$outdir" $BASE_FLAGS \
        --aic-metrics="$metric" --l2=on \
        2>&1 | tail -3 >&2 || echo "=== [WARN] $outdir failed ===" >&2
}

for B in 1 4 16; do
    build_fp16_om "$B"
    build_int8_om "$B"
    for PREC in fp16 int8; do
        run_msprof "$PREC" "$B" "ArithmeticUtilization" 20 3   # cube_util_fp16/int8_ratio
        run_msprof "$PREC" "$B" "PipeUtilization"        20 3   # mac/mte1/mte2/fixpipe
        run_msprof "$PREC" "$B" "L2Cache"                20 3   # LLC 命中率随精度变化
    done
done

echo "=== ALL DONE [$(date +%H:%M:%S)] ===" >&2
for d in msprof_et_bert_{fp16,int8}_b*_*; do
    [[ -d "$d" ]] || continue
    echo "  $d : op_summary=$(find "$d" -name 'op_summary*.csv' 2>/dev/null | wc -l)"
done

# ─────────────────────────────────────────────────────────────────────────
# 解析与对照(取回后本地)：
#   python3 benchmark/parse_msprof.py --glob 'msprof_et_bert_*_ArithmeticUtilization' \
#       --out data/calibration/etbert_int8_vs_fp16.json
#   预期对照(INT8 vs FP16,同 batch)：
#     ┌──────────────────────┬──────── FP16 ───────┬──────── INT8 ────────┐
#     │ cube_util_int8_ratio │ 0                    │ > 0(主要算力落 INT8) │
#     │ aic_pipes.mte2(HBM)  │ 基准                 │ ≈ 0.5×(权重位宽减半)  │
#     │ wall_clock           │ 基准                 │ 小 batch 几乎不变(host)│
#     │ L2 命中率            │ 基准                 │ 权重变小→命中率或上升 │
#   该对照若成立,即用**910B 实测**坐实“精度↓→访存占比↑、CUBE 更易空闲”的方向,
#   从而给 FP8/FP4(无硅)的 PRISM 外推背书。结果填入
#   docs/findings/etbert_fp8_fp4_910b_analysis.md 的“实测桥”一节。
