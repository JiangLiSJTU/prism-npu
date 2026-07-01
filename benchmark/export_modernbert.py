"""
ModernBERT-base ONNX 导出脚本（encoder, prefill S=4096）
模型: answerdotai/ModernBERT-base (149M params, 22 layers, GLU FFN)

用法:
  python benchmark/export_modernbert.py --S 4096 --batch 1
  python benchmark/export_modernbert.py --S 4096 --batch 1 4  # prefill with batch

输出: models/modernbert_base_prefill_S{S}_b{B}.onnx

依赖: pip install torch transformers>=4.48 onnx
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


def export_modernbert(S: int, batches: list[int], output_dir: Path):
    print(f"\n=== ModernBERT-base prefill (S={S}) ===")
    try:
        from transformers import AutoModel  # ModernBertModel available from 4.48+
    except ImportError:
        print("  need: pip install transformers>=4.48")
        return

    model_id = "answerdotai/ModernBERT-base"
    print(f"  downloading {model_id} ...")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=False, attn_implementation="eager")
    model.eval()
    model.float()  # FP32 for ONNX export; atc quantizes to FP16

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {total_params:,} ({total_params/1e6:.1f}M, ~{total_params*2/1e6:.0f} MB FP16)")

    for B in batches:
        dummy_ids = torch.zeros(B, S, dtype=torch.long)
        dummy_mask = torch.ones(B, S, dtype=torch.long)

        out_path = output_dir / f"modernbert_base_prefill_S{S}_b{B}.onnx"
        print(f"  exporting B={B} -> {out_path.name} ...")

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_ids, dummy_mask),
                str(out_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "last_hidden_state": {0: "batch", 1: "sequence"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
        size_mb = out_path.stat().st_size / 1e6
        print(f"    done: {size_mb:.1f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ModernBERT ONNX export")
    p.add_argument("--S", type=int, default=4096, help="sequence length (default: 4096)")
    p.add_argument("--batch", type=int, nargs="+", default=[1], help="batch sizes (default: 1)")
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = p.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_modernbert(args.S, args.batch, OUTPUT_DIR)
