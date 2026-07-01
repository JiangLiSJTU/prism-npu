"""
Qwen3-0.6B autoregressive decode 模式 ONNX 导出（小型 hypothesis 验证）

派生自 export_qwen3_prefill.py，关键差别：
  - input_ids: (B, 1) — 单 token 输入
  - past_key_values: 28 layers × (k, v) 对，每个 shape (B, 8, S_kv, 128) FP16
  - use_cache=True
  - cache_position 显式传入 = S_kv（next token position）

用途：仅做 1 个 hypothesis 验证配置（S_kv=128, B=1），跑 msprof 4 metrics
确认 decode 是 HBM-BW-bound（KV cache reload 主导）。

用法：
  cd ~/sim-experiment
  python benchmark/export_qwen3_decode.py --S_kv 128 --batch 1

输出：models/qwen3_06b_decode_Skv{S_kv}_b{B}.onnx
"""

import argparse
import sys
from pathlib import Path

import torch


OUTPUT_DIR = Path(__file__).parent.parent / "models"

# Qwen3-0.6B 架构常量（用于 dummy past_kv 形状）
NUM_LAYERS = 28
NUM_KV_HEADS = 8       # GQA：8 KV heads（vs 16 Q heads）
HEAD_DIM = 128


# ─────────────────────────────────────────────────────────────
# Patches（沿用 prefill 已验证的 3 个补丁）
# ─────────────────────────────────────────────────────────────
def _patch_qwen3_masking():
    """修复 transformers 5.x masking_utils.sdpa_mask 在 0-dim tensor 时的 BC 崩溃。"""
    try:
        import transformers.masking_utils as mu
        if not hasattr(mu, "sdpa_mask"):
            return
        _orig = mu.sdpa_mask

        def _safe_sdpa_mask(*args, **kwargs):
            args = list(args)
            for idx in (1, 2):
                if (idx < len(args)
                        and isinstance(args[idx], torch.Tensor)
                        and args[idx].dim() == 0):
                    args[idx] = int(args[idx].item())
            for key in ("q_length", "kv_length", "q_offset"):
                if (key in kwargs
                        and isinstance(kwargs[key], torch.Tensor)
                        and kwargs[key].dim() == 0):
                    kwargs[key] = int(kwargs[key].item())
            return _orig(*args, **kwargs)

        mu.sdpa_mask = _safe_sdpa_mask
        print("  [patch] masking_utils.sdpa_mask: 0-dim tensor BC 修复已应用")
    except Exception as exc:
        print(f"  [patch] masking_utils 补丁跳过：{exc}")


def _patch_torch_diff_for_onnx():
    """torch.diff 无 ONNX symbolic，用 narrow+sub 替代。"""
    _orig_diff = torch.diff

    def _onnx_safe_diff(input, n=1, dim=-1, prepend=None, append=None):
        parts = []
        if prepend is not None:
            parts.append(prepend)
        parts.append(input)
        if append is not None:
            parts.append(append)
        if len(parts) > 1:
            input = torch.cat(parts, dim=dim)
        result = input
        for _ in range(n):
            seq = result.shape[dim]
            lo = torch.narrow(result, dim, 0, seq - 1)
            hi = torch.narrow(result, dim, 1, seq - 1)
            result = hi - lo
        return result

    torch.diff = _onnx_safe_diff
    print("  [patch] torch.diff: ONNX Slice+Sub 替代已应用")


def _patch_cumsum_bool_cast():
    """Cast bool tensor → int32 before cumsum（ATC 不支持 Cumsum<DT_BOOL>）"""
    _orig_cumsum = torch.Tensor.cumsum

    def _safe_cumsum(self, *args, **kwargs):
        if self.dtype == torch.bool:
            self = self.to(torch.int32)
        return _orig_cumsum(self, *args, **kwargs)

    torch.Tensor.cumsum = _safe_cumsum
    print("  [patch] Tensor.cumsum: bool→int32 auto-cast（ATC 兼容）已应用")


# ─────────────────────────────────────────────────────────────
# Decode wrapper：input_ids + 56 个 past_kv 张量 → logits
# ─────────────────────────────────────────────────────────────
class DecodeWrapper(torch.nn.Module):
    """
    包装为可 ONNX trace 的 forward(input_ids, *past_kv_flat) 形式。

    past_kv_flat 顺序：[past_k_0, past_v_0, past_k_1, past_v_1, ..., past_k_27, past_v_27]
    每个张量 shape: (B, NUM_KV_HEADS, S_kv, HEAD_DIM)，FP32（torch 默认；ATC 转 OM 时降 FP16）
    """
    def __init__(self, hf_model, num_layers: int, s_kv: int):
        super().__init__()
        self.model = hf_model
        self.num_layers = num_layers
        self.s_kv = s_kv

    def forward(self, input_ids, *past_kv_flat):
        # 1) 重组为 (k, v) tuple list 格式（transformers 5.7 ddp_cache_data 接口）
        ddp_data = []
        for i in range(self.num_layers):
            k = past_kv_flat[2 * i]
            v = past_kv_flat[2 * i + 1]
            ddp_data.append((k, v))

        # 2) 用 ddp_cache_data 初始化 DynamicCache（transformers 5.7+）
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache(ddp_cache_data=ddp_data)

        # 3) cache_position：next token 位置 = S_kv
        cache_position = torch.arange(
            self.s_kv, self.s_kv + input_ids.shape[1],
            dtype=torch.long, device=input_ids.device
        )

        out = self.model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        return out.logits


# ─────────────────────────────────────────────────────────────
# 主导出函数
# ─────────────────────────────────────────────────────────────
def export_qwen3_decode(B: int, S_kv: int, output_dir: Path):
    print(f"\n=== Qwen3-0.6B decode mode export (B={B}, S_kv={S_kv}) ===")
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("  ✗ 需要 transformers >= 4.51.0")
        return False

    _patch_qwen3_masking()
    _patch_torch_diff_for_onnx()
    _patch_cumsum_bool_cast()

    print(f"  从 HuggingFace 加载 Qwen3-0.6B 权重...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    base_model.eval()
    model = DecodeWrapper(base_model, num_layers=NUM_LAYERS, s_kv=S_kv)

    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  参数量：{total_params:,}  ({total_params / 1e6:.1f}M)")

    # Dummy 输入
    dummy_input_ids = torch.zeros(B, 1, dtype=torch.long)
    # 56 个 dummy past_kv 张量，每个 (B, 8, S_kv, 128)
    past_kv_shape = (B, NUM_KV_HEADS, S_kv, HEAD_DIM)
    dummy_past = []
    for layer in range(NUM_LAYERS):
        dummy_past.append(torch.zeros(*past_kv_shape, dtype=torch.float32))   # k
        dummy_past.append(torch.zeros(*past_kv_shape, dtype=torch.float32))   # v

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"qwen3_06b_decode_Skv{S_kv}_b{B}.onnx"

    # Input names: input_ids + past_k/v_{i}
    input_names = ["input_ids"]
    for i in range(NUM_LAYERS):
        input_names.append(f"past_k_{i}")
        input_names.append(f"past_v_{i}")

    print(f"\n  导出 → {out_path.name}")
    print(f"  past_kv 总占用：{NUM_LAYERS * 2 * B * NUM_KV_HEADS * S_kv * HEAD_DIM * 4 / 1e6:.1f} MB (FP32 dummy)")
    print(f"  ONNX 输入数：{len(input_names)}（1 input_ids + {NUM_LAYERS * 2} past_kv）")

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input_ids, *dummy_past),
                str(out_path),
                input_names=input_names,
                output_names=["logits"],
                opset_version=14,
                do_constant_folding=True,
            )
        file_size_mb = out_path.stat().st_size / 1e6
        print(f"  ✓ B={B} S_kv={S_kv}  size={file_size_mb:.1f} MB")
        return True
    except Exception as exc:
        print(f"  ✗ 导出失败：{exc}")
        if out_path.exists() and out_path.stat().st_size < 100_000:
            out_path.unlink()
        import traceback; traceback.print_exc()
        return False


def main():
    p = argparse.ArgumentParser(description="Qwen3-0.6B decode 模式 ONNX 导出")
    p.add_argument("--S_kv", type=int, required=True, help="过去 KV cache 长度 (例如 128)")
    p.add_argument("--batch", type=int, default=1, help="batch size（默认 1）")
    p.add_argument("--output_dir", default=str(OUTPUT_DIR), help="输出目录")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    print(f"输出目录：{output_dir}")
    print(f"S_kv = {args.S_kv}, batch = {args.batch}")

    success = export_qwen3_decode(args.batch, args.S_kv, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
