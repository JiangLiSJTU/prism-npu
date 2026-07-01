"""
Qwen3-0.6B prefill 模式 ONNX 导出（参数化 S 与 B）

派生自 export_bert_gpt2_qwen.py 的 export_qwen3_06b 函数，
将 hardcoded S=512 改为参数化，支持 prefill 多档：S=[256, 512, 4096, 8192]。

用法（在 910B4 服务器上运行）：
  cd ~/sim-experiment
  python benchmark/export_qwen3_prefill.py --S 256 --batch 1
  python benchmark/export_qwen3_prefill.py --S 512 --batch 1 4
  python benchmark/export_qwen3_prefill.py --S 4096 --batch 1
  python benchmark/export_qwen3_prefill.py --S 8192 --batch 1

输出文件命名：models/qwen3_06b_prefill_S{S}_b{B}.onnx

依赖：
  - transformers>=4.51.0
  - torch (PyTorch 2.5+)
  - HuggingFace 缓存中已有 Qwen/Qwen3-0.6B 权重

⚠️ 内存：S=8192 B=1 单层 attention scores ≈ 2GB；OK
  S=8192 B=4 ≈ 8GB / layer；可能需要 --low_cpu_mem_usage
"""

import argparse
import sys
from pathlib import Path

import torch


OUTPUT_DIR = Path(__file__).parent.parent / "models"


# ─────────────────────────────────────────────────────────────
# Patches（沿用 export_bert_gpt2_qwen.py 中已验证的两个补丁）
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
    """
    Cast bool tensor → int32 before cumsum.
    根因：transformers/masking_utils.py:1002 在 create_chunked_causal_mask 内
    对 bool attention_mask 直接 .cumsum(-1)，ATC 不支持 Cumsum<DT_BOOL>。
    Patch：monkey-patch torch.Tensor.cumsum 自动 cast bool → int32。
    """
    _orig_cumsum = torch.Tensor.cumsum

    def _safe_cumsum(self, *args, **kwargs):
        if self.dtype == torch.bool:
            self = self.to(torch.int32)
        return _orig_cumsum(self, *args, **kwargs)

    torch.Tensor.cumsum = _safe_cumsum
    print("  [patch] Tensor.cumsum: bool→int32 auto-cast（ATC 兼容）已应用")


# ─────────────────────────────────────────────────────────────
# Wrapper（强制 use_cache=False + 仅返回 logits）
# ─────────────────────────────────────────────────────────────
class CausalLMWrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids):
        out = self.model(input_ids=input_ids, use_cache=False)
        return out.logits


# ─────────────────────────────────────────────────────────────
# 主导出函数（参数化 S）
# ─────────────────────────────────────────────────────────────
def export_qwen3_prefill(batches, S, output_dir: Path):
    print(f"\n=== Qwen3-0.6B prefill mode export (S={S}) ===")
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("  ✗ 需要 transformers >= 4.51.0")
        return False

    _patch_qwen3_masking()
    _patch_torch_diff_for_onnx()
    _patch_cumsum_bool_cast()

    print(f"  从 HuggingFace 加载 Qwen3-0.6B 权重（缓存已有）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    base_model.eval()
    model = CausalLMWrapper(base_model)

    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  参数量：{total_params:,}  ({total_params / 1e6:.1f}M)")

    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for B in batches:
        out_path = output_dir / f"qwen3_06b_prefill_S{S}_b{B}.onnx"
        print(f"\n  导出 S={S} B={B} → {out_path.name}")
        print(f"  （activation memory 估算：{B * S * 1024 * 28 * 2 / 1e9:.2f} GB；attention scores per layer：{B * 16 * S * S * 2 / 1e9:.2f} GB）")

        dummy_ids = torch.zeros(B, S, dtype=torch.long)

        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_ids,
                    str(out_path),
                    input_names=["input_ids"],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids": {0: "batch"},
                        "logits": {0: "batch"},
                    },
                    opset_version=14,
                    do_constant_folding=True,
                )
            file_size_mb = out_path.stat().st_size / 1e6
            print(f"  ✓ S={S} B={B}  size={file_size_mb:.1f} MB")
            success_count += 1
        except Exception as exc:
            print(f"  ✗ S={S} B={B} 导出失败：{exc}")
            if out_path.exists() and out_path.stat().st_size < 100_000:
                out_path.unlink()
            continue

    print(f"\n  完成：{success_count} / {len(batches)} 成功")
    return success_count > 0


def main():
    p = argparse.ArgumentParser(description="Qwen3-0.6B prefill 模式 ONNX 导出（参数化 S）")
    p.add_argument("--S", type=int, required=True, help="序列长度（256 / 512 / 4096 / 8192 等）")
    p.add_argument("--batch", nargs="+", type=int, default=[1], help="batch size 列表（默认 [1]）")
    p.add_argument("--output_dir", default=str(OUTPUT_DIR), help="输出目录")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    print(f"输出目录：{output_dir}")
    print(f"S = {args.S}, batches = {args.batch}")

    success = export_qwen3_prefill(args.batch, args.S, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
