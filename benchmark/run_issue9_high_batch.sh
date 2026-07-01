#!/bin/bash
# Issue #9 Phase 1b — High-batch prefill data collection
# 在 Ascend 910B4 NPU 上批量做 ONNX export → ATC → ais_bench+msprof PipeUtilization
# 目的：采 v8 训练集未见的 high-batch (B≥16) 工况，检验 AIV_BOUND classifier 误判
#
# 使用：
#   1. 把本脚本与 PRISM 仓库一起 rsync 到 NPU server (e.g. <user>@<npu_host>:~/sim-experiment/)
#   2. ssh 上去，bash benchmark/run_issue9_high_batch.sh 2>&1 | tee /tmp/issue9_run.log
#   3. 跑完 rsync msprof_data/msprof_issue9_*_PipeUtilization/ 回本地
#
# 预计 NPU 时间：~30-50 min 全部 5 个 config（含 ATC 编译 + msprof loop=5）
set -euo pipefail

cd ~/sim-experiment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate dev_torch25 2>/dev/null || true
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ─────────────────────────────────────────────────────────────────────
# Thread caps — NPU box has 192 cores but OpenBLAS precompiled NUM_THREADS=128.
# Without these env vars, large-batch ONNX trace overruns the auxiliary-array
# fallback and segfaults during export (observed: B=16 segfault, B=32 hang).
# 32 is plenty for ONNX export (mostly serial tracing) and well below the cap.
# ─────────────────────────────────────────────────────────────────────
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

# ─────────────────────────────────────────────────────────────────────
# Configs to collect — matches issue-9 §4 Phase 1 plan
# (export_script, S, batch, expected_classification)
# ─────────────────────────────────────────────────────────────────────
CONFIGS=(
  "qwen3_prefill_sdpa 512  16  boundary"          # Qwen3-0.6B, B=16
  "qwen3_prefill_sdpa 512  32  aic_compute"       # Qwen3-0.6B, B=32
  "qwen3_prefill_sdpa 256  64  aic_compute"       # Qwen3-0.6B, B=64 (high B, lower S)
  "llama_3_2_1b_prefill_sdpa 2048 8  boundary"    # Llama-1B, B=8 (current TRAIN edge)
  "llama_3_2_1b_prefill_sdpa 2048 16 aic_compute" # Llama-1B, B=16
)

# Map short script name → full file + stem prefix
get_export_script() {
  case "$1" in
    qwen3_prefill_sdpa)        echo "benchmark/export_qwen3_prefill_sdpa.py" ;;
    llama_3_2_1b_prefill_sdpa) echo "benchmark/export_llama_3_2_1b_prefill_sdpa.py" ;;
    *) echo ""; return 1 ;;
  esac
}

get_stem_prefix() {
  case "$1" in
    qwen3_prefill_sdpa)        echo "qwen3_06b_prefill" ;;
    llama_3_2_1b_prefill_sdpa) echo "llama_3_2_1b_prefill" ;;
    *) echo ""; return 1 ;;
  esac
}

LOOP=5         # msprof iterations (high-batch single inference ~ seconds; 5 is enough for stable mean)
WARMUP=1
MSPROF_ROOT="msprof_data"
mkdir -p models om "$MSPROF_ROOT"

# ─────────────────────────────────────────────────────────────────────
# Pipeline per config
# ─────────────────────────────────────────────────────────────────────
run_one_config() {
  local script="$1" S="$2" B="$3" tag="$4"
  local export_py
  export_py=$(get_export_script "$script") || { echo "✗ unknown script: $script"; return 1; }
  local stem_prefix
  stem_prefix=$(get_stem_prefix "$script") || return 1
  local stem="${stem_prefix}_S${S}_b${B}_sdpa"
  local issue9_stem="issue9_${stem}"
  local onnx="models/${stem}.onnx"
  local om="om/${stem}.om"
  local outdir="${MSPROF_ROOT}/msprof_${issue9_stem}_PipeUtilization"

  echo
  echo "═══════════════════════════════════════════════════════════════"
  echo "[$(date +%H:%M:%S)] CONFIG: ${stem}   (expected: ${tag})"
  echo "═══════════════════════════════════════════════════════════════"

  # 1. ONNX export (skip if exists; --S --batch unified across export scripts)
  if [ ! -f "$onnx" ]; then
    echo "→ exporting ONNX (S=${S} batch=${B})"
    python3 "$export_py" --S "$S" --batch "$B" || { echo "✗ ONNX export failed"; return 1; }
  else
    echo "→ ONNX exists, skipping export: $onnx"
  fi

  # 2. ATC convert
  if [ ! -f "$om" ]; then
    echo "→ ATC compile"
    local atc_log="/tmp/atc_${issue9_stem}.log"
    atc --model="$onnx" --framework=5 \
        --output="om/${stem}" --soc_version=Ascend910B4 \
        --input_shape="input_ids:${B},${S};attention_mask:${B},${S}" \
        --precision_mode=allow_fp32_to_fp16 \
        --log=info \
        > "$atc_log" 2>&1 && echo "  ATC OK" || { echo "✗ ATC failed (see $atc_log)"; tail -20 "$atc_log"; return 1; }
  else
    echo "→ OM exists, skipping ATC: $om"
  fi

  # 3. msprof — chmod 750 to avoid group-writable rejection (Issue #5 lesson)
  rm -rf "$outdir"
  mkdir -p "$outdir"
  chmod 750 "$outdir"

  echo "→ msprof PipeUtilization (loop=${LOOP} warmup=${WARMUP})"
  msprof --application="python3 -m ais_bench --model $om --loop $LOOP --warmup_count $WARMUP" \
         --output="./$outdir" \
         --task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off \
         --aic-metrics=PipeUtilization --l2=on --analyze=on \
         > "/tmp/msprof_${issue9_stem}.log" 2>&1 && echo "  msprof OK" || { echo "✗ msprof failed"; tail -10 "/tmp/msprof_${issue9_stem}.log"; return 1; }

  # 4. Verify products + brief sanity
  local csv step
  csv=$(find "$outdir" -name 'op_summary*.csv' 2>/dev/null | head -1)
  step=$(find "$outdir" -name 'step_trace*.csv' 2>/dev/null | head -1)
  if [ -n "$csv" ]; then
    echo "  ✓ op_summary: $(wc -l < "$csv") rows"
  else
    echo "  ✗ op_summary MISSING"
    return 1
  fi
  if [ -n "$step" ]; then
    local mean_us
    mean_us=$(awk -F',' 'NR>1 && $0~/[0-9]/ {s+=$2; n++} END{if(n>0) printf "%.1f", s/n}' "$step")
    echo "  ✓ step_trace: iter mean ≈ ${mean_us} μs"
  fi

  return 0
}

# ─────────────────────────────────────────────────────────────────────
# Main loop with continue-on-error so one failure doesn't kill the rest
# ─────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "Issue #9 high-batch data collection — ${#CONFIGS[@]} configs"
echo "Start: $(date)"
echo "═══════════════════════════════════════════════════════════════"

passed=()
failed=()
for entry in "${CONFIGS[@]}"; do
  read -r script S B tag <<< "$entry"
  if run_one_config "$script" "$S" "$B" "$tag"; then
    passed+=("${script}_S${S}_b${B}")
  else
    failed+=("${script}_S${S}_b${B}")
    echo "⚠ continuing despite failure on ${script}_S${S}_b${B}"
  fi
done

echo
echo "═══════════════════════════════════════════════════════════════"
echo "End: $(date)"
echo "PASSED (${#passed[@]}): ${passed[*]}"
echo "FAILED (${#failed[@]}): ${failed[*]}"
echo "═══════════════════════════════════════════════════════════════"
echo
echo "Next: rsync to local"
echo "  rsync -avz <user>@<npu_host>:~/sim-experiment/msprof_data/msprof_issue9_* \\"
echo "       <repo>/msprof_data/"
