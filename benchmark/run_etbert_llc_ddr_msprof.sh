#!/bin/bash
# ET-BERT — 补齐仓库三项空白之 ①LLC(L2)命中率 + ②DDR/HBM 带宽利用率
#
# 背景：data/calibration/ 目前只有 ArithmeticUtilization(cube_util) 与
#       PipeUtilization(pipe 分解)，**没有** L2/LLC 命中率，也**没有** DDR 带宽
#       利用率。本脚本用 msprof 的 L2Cache / Memory aic-metric + 系统级
#       sys-hardware-mem 采集这两类计数器，为 Q4(Cache 能否兜模型 / 加 Cache 是否
#       降带宽 / DDR 利用率)提供**实测**依据（而非 PRISM 预测）。
#
# 采集对象：ET-BERT(BERT-base 12 层 + vocab=256) FP16，batch = 1/4/8/16
#   注：910B 上 L2(96MB)即 LLC；本脚本测的是 L2 命中率与 HBM2e(392GB/s)利用率。
#
# 每个 batch 跑两遍 msprof(aic-metrics 一次只能选一组)：
#   pass-1  L2Cache  → 每算子 l2_cache_hit_rate / main_mem_read|write_bw
#   pass-2  Memory   → 每算子 read/write bytes + 各 pipe 带宽
# 并全程开 --sys-hardware-mem=on 采系统级 DDR/HBM 读写带宽时间序列。
#
# 前置条件(910B 真机)：
#   1) om/et_bert_b{1,4,8,16}.om 齐全(见 benchmark/convert_onnx_to_om.sh)
#   2) CANN 8.5 + ais_bench + msprof 就绪
#   3) msprof 支持 --aic-metrics=L2Cache 与 --sys-hardware-mem(CANN≥7.0)
#
# 运行：
#   ssh user@npu-server
#   cd ~/sim-experiment
#   bash benchmark/run_etbert_llc_ddr_msprof.sh 2>&1 | tee etbert_llc_ddr.log
#
# 取回：
#   rsync -avz user@npu-server:~/sim-experiment/msprof_et_bert_b*_{L2Cache,Memory}/ msprof_data/
#
# 解析(取回后本地跑)：
#   python3 benchmark/parse_msprof.py --glob 'msprof_et_bert_b*_L2Cache' \
#       --fields l2_cache_hit_rate,main_mem_read_bw,main_mem_write_bw \
#       --out data/calibration/etbert_llc_extracted.json
#   # DDR 利用率 = (ddr_read_bw + ddr_write_bw) / 392  (GB/s ÷ HBM2e 峰值)
#   # 见脚本末尾“解析与口径”注释。

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dev_torch25
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/sim-experiment

# task-time/ai-core/aicpu/runtime-api 与既有脚本一致；额外开系统级 DDR 带宽。
BASE_FLAGS="--task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off --sys-hardware-mem=on"

run_pass() {
    local batch=$1; local metric=$2; local loop=$3; local warmup=$4
    local omfile="om/et_bert_b${batch}.om"
    local outdir="msprof_et_bert_b${batch}_${metric}"

    if [[ ! -f "$omfile" ]]; then
        echo "=== [SKIP] $omfile not found ===" >&2; return
    fi
    rm -rf "$outdir"; mkdir -p "$outdir"; chmod 750 "$outdir"
    echo "=== [$(date +%H:%M:%S)] $outdir (metric=$metric loop=$loop warmup=$warmup) ===" >&2

    msprof --application="python3 -m ais_bench --model $omfile --loop $loop --warmup_count $warmup" \
        --output="./$outdir" $BASE_FLAGS \
        --aic-metrics="$metric" --l2=on \
        2>&1 | tail -3 >&2 || echo "=== [WARN] $outdir failed ===" >&2
    echo "=== [$(date +%H:%M:%S)] Done $outdir ===" >&2
}

for B in 1 4 8 16; do
    run_pass "$B" "L2Cache" 20 3   # ①LLC/L2 命中率
    run_pass "$B" "Memory"  20 3   # ②各 pipe 读写带宽(配合系统 DDR 带宽算利用率)
done

echo "=== ALL DONE [$(date +%H:%M:%S)] ===" >&2
echo "=== 检查每个目录的 op_summary / 系统内存 CSV ==="
for d in msprof_et_bert_b*_{L2Cache,Memory}; do
    [[ -d "$d" ]] || continue
    op=$(find "$d" -name "op_summary*.csv" 2>/dev/null | wc -l)
    mem=$(find "$d" -iname "*ddr*.csv" -o -iname "*hbm*.csv" -o -iname "*hardware_mem*.csv" 2>/dev/null | wc -l)
    echo "  $d : op_summary=$op  sys_mem_csv=$mem"
done

# ─────────────────────────────────────────────────────────────────────────
# 解析与口径(取回后本地)：
#   • LLC/L2 命中率：op_summary*.csv 的 l2_cache_hit_rate(逐算子，按 aicore_time 加权)
#       → 整网 LLC 命中率 = Σ(hit_rate × aicore_time) / Σ(aicore_time)
#   • DDR/HBM 带宽利用率：sys-hardware-mem 的 ddr_read_bw + ddr_write_bw(GB/s)
#       → 利用率 = 平均(read_bw+write_bw) / 392(HBM2e 峰值)
#       → 也可用 Memory pass 的 main_mem_read/write_bw 交叉校验
#   • 预期(用于校验 PRISM 外推方向)：
#       - ET-BERT 权重 170MB > L2 96MB → 权重几乎无复用,L2 命中率应偏低(<~50%)
#       - b=1 host 受限,DDR 利用率低(~15-20%);b≥16 才接近饱和
#   这三个数填入后,docs/findings/etbert_fp8_fp4_910b_analysis.md 的 Q4 表即由
#   “PRISM 预测”升级为“910B 实测”。
