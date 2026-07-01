"""
SmolLM2-135M decode ONNX 导出脚本（decoder-only, M=1 single token）
模型: HuggingFaceTB/SmolLM2-135M (135M params, 30 layers, GQA 9Q/3KV)

用法:
  python benchmark/export_smollm2_decode.py --batch 1

输出: models/smollm2_135m_decode_b{B}.onnx

依赖: pip install torch transformers>=4.46 onnx
"""

import argparse
from pathlib import Path
import torch

# Monkey-patch transformers.masking_utils.sdpa_mask to handle 0-d q_length tensors
# (workaround for ONNX export bug in transformers 5.7+: q_length.shape[0] fails
# when q_length is a 0-d scalar tensor)
def _patch_sdpa_mask():
    import torch
    import transformers.masking_utils as mu
    _orig = mu.sdpa_mask
    def _patched(*args, **kwargs):
        if "q_length" in kwargs:
            ql = kwargs["q_length"]
            if isinstance(ql, torch.Tensor) and ql.ndim == 0:
                kwargs["q_length"] = ql.unsqueeze(0)
        elif len(args) >= 2:
            arg_list = list(args)
            if isinstance(arg_list[1], torch.Tensor) and arg_list[1].ndim == 0:
                arg_list[1] = arg_list[1].unsqueeze(0)
                args = tuple(arg_list)
        return _orig(*args, **kwargs)
    mu.sdpa_mask = _patched
_patch_sdpa_mask()


OUTPUT_DIR = Path(__file__).parent.parent / "models"

# SmolLM2-135M architecture constants (for dummy KV cache shapes)
NUM_LAYERS = 30
NUM_KV_HEADS = 3   # GQA: 3 KV heads (vs 9 Q heads)
HEAD_DIM = 64
HIDDEN_SIZE = 576


def export_smollm2_decode(batches: list[int], output_dir: Path):
    print("\n=== SmolLM2-135M decode (S=1) ===")
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("  need: pip install transformers>=4.46")
        return

    model_id = "HuggingFaceTB/SmolLM2-135M"
    print(f"  downloading {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {total_params:,} ({total_params/1e6:.1f}M, ~{total_params*2/1e6:.0f} MB FP16)")

    # SmolLM2 uses Llama architecture under the hood
    for B in batches:
        dummy_ids = torch.zeros(B, 1, dtype=torch.long)

        out_path = output_dir / f"smollm2_135m_decode_b{B}.onnx"
        print(f"  exporting B={B} -> {out_path.name} ...")

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_ids,
                str(out_path),
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
        size_mb = out_path.stat().st_size / 1e6
        print(f"    done: {size_mb:.1f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SmolLM2-135M decode ONNX export")
    p.add_argument("--batch", type=int, nargs="+", default=[1], help="batch sizes (default: 1)")
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = p.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_smollm2_decode(args.batch, OUTPUT_DIR)
