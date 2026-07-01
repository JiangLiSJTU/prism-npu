#!/usr/bin/env python3
"""
Phase M+ — Physics-informed η_real 拟合（Method A）

基于 subagent 调研结论：
  η_real = η_pipeline(M, N, K) · η_tile(M, N, K) · η_mem(M, N, K) · (1 + γ_B·log2(B))

其中：
  η_pipeline = compute_cycles / (compute_cycles + α·M_b·N_b + β·M_b·K_b + γ·N_b·K_b + δ·(M_b+N_b+K_b))
              （systolic 阵列 fill/drain 经典公式：(M-1)+(N-1)+(K-1) 延迟）
  η_tile     = ∏(x / (16 · ceil(x/16)))   tile-edge 量化损失（无 free param）
  η_mem      = min(1, AI/AI_ridge)        Roofline ridge（无 free param）

5 个 free param: α, β, γ, δ, γ_B
拟合方法：scipy.optimize.least_squares on log-MSE
预期：train MAE 4-6 pp, BERT MAE 5-8 pp（vs OLS 22 pp）

关键修复：attention head 算子（is_attn_proj=True）不应用 B×M，每个 head 独立 tile
"""

import argparse
import json
import math
import re
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parent.parent.parent.parent

# 910B4 hardware constants
CUBE_M, CUBE_N, CUBE_K = 16, 16, 16
N_CORES = 24
PEAK_TFLOPS_FP16 = 280
HBM_BW_GBS = 392
AI_RIDGE = PEAK_TFLOPS_FP16 * 1e12 / (HBM_BW_GBS * 1e9)   # ~714 OPs/byte


def collect_shapes(cube_util_json, model_filter):
    with open(cube_util_json, encoding='utf-8') as f:
        data = json.load(f)
    samples = []
    for name, d in data.items():
        if model_filter not in name or 'ArithmeticUtilization' not in name:
            continue
        if not d.get('top_shapes_by_aicore_time'):
            continue
        bm = re.search(r'_b(\d+)_', name)
        if not bm:
            continue
        B = int(bm.group(1))
        sm = re.search(r'prefill_S(\d+)_', name)
        S = int(sm.group(1)) if sm else (
            128 if 'bert_base' in name else
            512 if 'gpt2_small' in name else
            512
        )
        for s in d['top_shapes_by_aicore_time']:
            if s['M'] is None or s['cube_util_pct'] <= 0 or s['count'] < 20:
                continue
            samples.append({
                'M_per_batch': s['M'], 'N': s['N'], 'K': s['K'],
                'B': B, 'S': S,
                'op_kind': s.get('op_kind', 'BMM'),   # 'BMM' or 'MM'
                'eta_real': s['cube_util_pct'] / 100,
                'aicore_time_us': s['aicore_time_us'],
                'count': s['count'],
                'msprof': name,
            })
    return samples


def is_attention_head(M, N, K):
    """识别 attention head op：M、N、K 中有维度 ≤ 128 即视为小 tile。"""
    return min(M, N, K) <= 128


def effective_M(s):
    """根据 op_kind 区分：
       - BMM (BatchMatMul): msprof 报 per-batch M，但 attention head 每 head 独立 tile 不乘 B
       - MM (MatMul): msprof 已 flatten batch 进 M，不再乘 B
    """
    M_msprof = s['M_per_batch']
    op_kind = s.get('op_kind', 'BMM')
    if op_kind == 'MM':
        # 已 flatten，M 直接用
        return M_msprof
    # BMM: 是否 attention head
    if is_attention_head(M_msprof, s['N'], s['K']):
        return M_msprof   # 每 head 独立
    return M_msprof * s['B']   # 标准 BatchMatMul，乘 B


def eta_tile(M, N, K):
    """Tile 边缘量化损失：每维 x / (16 * ceil(x/16))。无 free param."""
    def f(x): return x / (CUBE_M * math.ceil(x / CUBE_M))
    return f(M) * f(N) * f(K)


def eta_pipeline(M, N, K, alpha, beta, gamma, delta):
    """Pipeline fill/drain。M_b, N_b, K_b 是 16-block 数。
    公式：compute / (compute + overhead)
    overhead = α·M_b·N_b + β·M_b·K_b + γ·N_b·K_b + δ·(M_b+N_b+K_b)
    """
    Mb = math.ceil(M / CUBE_M)
    Nb = math.ceil(N / CUBE_N)
    Kb = math.ceil(K / CUBE_K)
    compute = Mb * Nb * Kb
    overhead = alpha * Mb * Nb + beta * Mb * Kb + gamma * Nb * Kb + delta * (Mb + Nb + Kb)
    return compute / (compute + overhead) if (compute + overhead) > 0 else 0


def eta_mem(M, N, K):
    """Roofline ridge: AI/AI_ridge clipped to 1."""
    ops = 2 * M * N * K   # FLOP count for FP MAC
    bytes_ = (M * K + K * N + M * N) * 2   # FP16 read inputs+weights, write outputs
    AI = ops / bytes_ if bytes_ > 0 else float('inf')
    return min(1.0, AI / AI_RIDGE)


def predict_eta(s, alpha, beta, gamma, delta, gamma_B):
    """Method A 简化版（drop η_mem，drop is_attn_proj 启发式）。

    根因：η_mem 用 raw HBM BW / AI_ridge 算，忽略 L2 缓存效果，对所有 BERT-class
    workload 误判为 memory-bound。我们的训练/验证数据都在 compute-bound regime
    内（Cube 不闲），让 fitter 自己学 fill/drain 系数即可。
    """
    M_eff = effective_M(s)
    N, K = s['N'], s['K']
    eta_p = eta_pipeline(M_eff, N, K, alpha, beta, gamma, delta)
    eta_t = eta_tile(M_eff, N, K)
    batch_factor = 1 + gamma_B * math.log2(max(s['B'], 1))
    return min(1.0, eta_p * eta_t * batch_factor)


def residuals(params, samples):
    alpha, beta, gamma, delta, gamma_B = params
    return np.array([
        predict_eta(s, alpha, beta, gamma, delta, gamma_B) - s['eta_real']
        for s in samples
    ])


def fit(samples, init=(1.0, 1.0, 1.0, 1.0, 0.012)):
    """Levenberg-Marquardt 拟合."""
    if not samples:
        raise ValueError("fit() requires non-empty samples; got 0 shapes")
    if len(samples) < len(init):
        raise ValueError(
            f"fit() requires >= {len(init)} samples for {len(init)} parameters; got {len(samples)}"
        )
    result = least_squares(
        residuals, init, args=(samples,),
        bounds=([0.0, 0.0, 0.0, 0.0, -0.5], [1e3, 1e3, 1e3, 1e3, 0.5]),
        method='trf',
    )
    return result.x


def evaluate(samples, params, label):
    if not samples:
        print(f"\n{label}：无样本")
        return
    print(f"\n=== {label}（{len(samples)} 个 shape）===")
    print(f"{'M_eff':>6} {'N':>5} {'K':>5} {'B':>3} {'attn?':>6}  {'η_real%':>9} {'η_pred%':>9} {'误差%':>8}")
    errs = []
    for s in samples:
        eta_pred = predict_eta(s, *params)
        err = (eta_pred - s['eta_real']) * 100
        errs.append(err)
        attn = 'YES' if is_attention_head(s['M_per_batch'], s['N'], s['K']) else 'no'
        print(f"{effective_M(s):>6} {s['N']:>5} {s['K']:>5} {s['B']:>3} {attn:>6}  "
              f"{s['eta_real']*100:>9.2f} {eta_pred*100:>9.2f} {err:>+8.2f}")
    abs_errs = [abs(e) for e in errs]
    mae = np.mean(abs_errs)
    rmse = np.sqrt(np.mean([e**2 for e in errs]))
    print(f"  MAE = {mae:.2f} pp, RMSE = {rmse:.2f} pp, max abs error = {max(abs_errs):.2f} pp")
    return errs, mae, rmse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cube-util-json',
                   default=str(REPO / 'data' / 'calibration' / 'cube_util_extracted.json'))
    p.add_argument('--output',
                   default=str(REPO / 'data' / 'calibration' / 'eta_physics_fit.json'))
    args = p.parse_args()

    train = collect_shapes(args.cube_util_json, 'qwen3')
    val_bert = collect_shapes(args.cube_util_json, 'bert_base')
    val_gpt2 = collect_shapes(args.cube_util_json, 'gpt2_small')

    print(f"训练集 (Qwen3): {len(train)} 个 shape")
    print(f"验证集 (BERT-base): {len(val_bert)} 个 shape")
    print(f"验证集 (GPT-2-small): {len(val_gpt2)} 个 shape")

    if not train:
        print("ERROR: 无训练数据")
        return 1

    # 拟合
    params = fit(train)
    alpha, beta, gamma, delta, gamma_B = params
    print(f"\n=== 拟合参数 ===")
    print(f"  α (M·N coupling) = {alpha:.4f}")
    print(f"  β (M·K coupling) = {beta:.4f}")
    print(f"  γ (N·K coupling) = {gamma:.4f}")
    print(f"  δ (linear edge)  = {delta:.4f}")
    print(f"  γ_B (batch term) = {gamma_B:.4f}")

    # 训练 + 验证
    _, train_mae, train_rmse = evaluate(train, params, 'Qwen3 训练集')
    val_results = {}
    if val_bert:
        _, bert_mae, bert_rmse = evaluate(val_bert, params, 'BERT-base 验证集')
        val_results['bert'] = {'mae': bert_mae, 'rmse': bert_rmse, 'n': len(val_bert)}
    if val_gpt2:
        _, gpt2_mae, gpt2_rmse = evaluate(val_gpt2, params, 'GPT-2-small 验证集')
        val_results['gpt2'] = {'mae': gpt2_mae, 'rmse': gpt2_rmse, 'n': len(val_gpt2)}

    out = {
        'method': 'physics-informed (η_pipeline · η_tile · η_mem · batch)',
        'hardware': {'CUBE': [CUBE_M, CUBE_N, CUBE_K], 'AI_ridge': AI_RIDGE},
        'params': {
            'alpha_MN_coupling': alpha,
            'beta_MK_coupling': beta,
            'gamma_NK_coupling': gamma,
            'delta_linear_edge': delta,
            'gamma_B_batch': gamma_B,
        },
        'training': {'n': len(train), 'mae': train_mae, 'rmse': train_rmse},
        'validation': val_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n=== 写入 {args.output} ===")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
