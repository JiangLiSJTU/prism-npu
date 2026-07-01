"""Qwen3-0.6B prefill ONNX export — SDPA attention path (Path B for FlashAttention).

Difference from eager export:
- attn_implementation="sdpa"  (vs "eager")
- SDPA generates F.scaled_dot_product_attention call, which ONNX export
  lowers to a single Attention/MultiHeadAttention op (opset 14+) instead
  of the MatMul→Softmax→MatMul triple.
- ATC's libops_all_onnx_plugin.so has `ge::op::PromptFlashAttention`
  recognized — fingers crossed it picks this up.

Outputs:  models/qwen3_06b_prefill_S{S}_b{B}_sdpa.onnx
"""
import argparse, sys
from pathlib import Path
import torch


OUTPUT_DIR = Path(__file__).parent.parent / "models"


def _patch_qwen3_masking():
    """Patch sdpa_mask in BOTH the module attribute AND the dispatch dict.

    transformers uses `ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]` for dispatch, so
    just patching mu.sdpa_mask is insufficient.
    """
    try:
        import transformers.masking_utils as mu
        if not hasattr(mu, "sdpa_mask"): return
        _orig = mu.sdpa_mask

        def _safe(*args, **kwargs):
            # sdpa_mask signature: (batch_size, q_length, kv_length, q_offset=0, kv_offset=0, ...)
            args = list(args)
            for idx in (1, 2, 3, 4):   # q_length, kv_length, q_offset, kv_offset
                if idx < len(args) and isinstance(args[idx], torch.Tensor) and args[idx].dim() == 0:
                    args[idx] = int(args[idx].item())
            for key in ("q_length", "kv_length", "q_offset", "kv_offset", "batch_size"):
                if key in kwargs and isinstance(kwargs[key], torch.Tensor) and kwargs[key].dim() == 0:
                    kwargs[key] = int(kwargs[key].item())
            return _orig(*args, **kwargs)

        # Patch the module attribute
        mu.sdpa_mask = _safe
        # Also patch the dispatch dict (this is what create_causal_mask actually uses)
        if hasattr(mu, "ALL_MASK_ATTENTION_FUNCTIONS"):
            mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = _safe
            print(f"  [patch] sdpa_mask + ALL_MASK_ATTENTION_FUNCTIONS['sdpa']")
        else:
            print(f"  [patch] sdpa_mask only (no dispatch dict found)")
    except Exception as e:
        print(f"  [patch] skip: {e}")


def _patch_torch_diff():
    _orig = torch.diff
    def _safe(input, n=1, dim=-1, prepend=None, append=None):
        parts = []
        if prepend is not None: parts.append(prepend)
        parts.append(input)
        if append is not None: parts.append(append)
        if len(parts) > 1: input = torch.cat(parts, dim=dim)
        result = input
        for _ in range(n):
            seq = result.shape[dim]
            lo = torch.narrow(result, dim, 0, seq - 1)
            hi = torch.narrow(result, dim, 1, seq - 1)
            result = hi - lo
        return result
    torch.diff = _safe
    print("  [patch] torch.diff")


def _patch_cumsum():
    _orig = torch.Tensor.cumsum
    def _safe(self, *args, **kwargs):
        if self.dtype == torch.bool: self = self.to(torch.int32)
        return _orig(self, *args, **kwargs)
    torch.Tensor.cumsum = _safe
    print("  [patch] cumsum")


class CausalLMWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          use_cache=False).logits


def export(S, batch, output_dir):
    print(f"\n=== Qwen3-0.6B prefill SDPA export (S={S} b={batch}) ===")
    from transformers import AutoModelForCausalLM
    _patch_qwen3_masking(); _patch_torch_diff(); _patch_cumsum()
    print(f"  loading Qwen/Qwen3-0.6B with attn_implementation='sdpa' ...")
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        attn_implementation="sdpa",   # ← KEY CHANGE
        torch_dtype=torch.float32,
    )
    base.eval()
    model = CausalLMWrapper(base)
    params = sum(p.numel() for p in base.parameters())
    print(f"  params: {params:,} ({params/1e6:.1f}M)")
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"qwen3_06b_prefill_S{S}_b{batch}_sdpa.onnx"
    print(f"  exporting → {out.name} (opset=17 for better SDPA support)")
    dummy_ids = torch.zeros(batch, S, dtype=torch.long)
    dummy_mask = torch.ones(batch, S, dtype=torch.long)   # full attention (no padding)
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_ids, dummy_mask),
                str(out),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids":      {0: "batch"},
                    "attention_mask": {0: "batch"},
                    "logits":         {0: "batch"},
                },
                opset_version=17,   # opset 17 lowers SDPA to Attention op natively
                do_constant_folding=True,
            )
        sz = out.stat().st_size / 1e6
        print(f"  ✓ {sz:.1f} MB → {out}")
        return True
    except Exception as e:
        import traceback
        print(f"  ✗ export failed: {e}")
        traceback.print_exc()
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--S", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = p.parse_args()
    sys.exit(0 if export(args.S, args.batch, Path(args.output_dir)) else 1)


if __name__ == "__main__":
    main()
