#!/bin/bash
# msprof 采集: ModernBERT S=4096 + SmolLM2 decode (910B4, 4 metrics each)
# Run on Ascend 910B4 server after ONNX->OM conversion.
#
# Usage: bash benchmark/run_new_models_msprof.sh <om_dir> <output_dir>
#   Default om_dir=om, output_dir=msprof_data

OM_DIR="${1:-om}"
OUT_DIR="${2:-msprof_data}"
SETUP='source ~/miniconda3/etc/profile.d/conda.sh && conda activate dev_torch25 && source /usr/local/Ascend/ascend-toolkit/set_env.sh'
BASE_FLAGS="--task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off"

mkdir -p "$OUT_DIR"

# ── ModernBERT S=4096 B=1 ──────────────────────────────────────
MB_OM="$OM_DIR/modernbert_base_prefill_S4096_b1.om"
if [ -f "$MB_OM" ]; then
    for metric in PipeUtilization ArithmeticUtilization L2Cache Memory; do
        DIR="$OUT_DIR/msprof_modernbert_S4096_b1_${metric}"
        echo "=== ModernBERT S=4096 b=1 $metric ==="
        bash -lc "$SETUP && cd ~/sim-experiment && rm -rf $DIR && \
          msprof --application=\"python3 -m ais_bench --model $MB_OM --loop 5 --warmup_count 1\" \
            --output=$DIR $BASE_FLAGS --aic-metrics=$metric --l2=on --analyze=on"
    done
else
    echo "WARN: $MB_OM not found, skip ModernBERT"
fi

# ── SmolLM2-135M decode B=1 ─────────────────────────────────────
SM_OM="$OM_DIR/smollm2_135m_decode_b1.om"
if [ -f "$SM_OM" ]; then
    for metric in PipeUtilization ArithmeticUtilization L2Cache Memory; do
        DIR="$OUT_DIR/msprof_smollm2_decode_b1_${metric}"
        echo "=== SmolLM2 decode b=1 $metric ==="
        bash -lc "$SETUP && cd ~/sim-experiment && rm -rf $DIR && \
          msprof --application=\"python3 -m ais_bench --model $SM_OM --loop 30 --warmup_count 5\" \
            --output=$DIR $BASE_FLAGS --aic-metrics=$metric --l2=on --analyze=on"
    done
else
    echo "WARN: $SM_OM not found, skip SmolLM2"
fi

echo "=== Done. Pull results with: ==="
echo "rsync -avz --include='*.csv' --include='*/' --exclude='*' \\
      npu-server:~/sim-experiment/$OUT_DIR/ \\
      ./msprof_data/"
