"""
BERT-base / GPT-2-small / Qwen3-0.6B  ONNX 导出脚本
从 HuggingFace 下载三个校准模型并批量导出为 ONNX，
再通过 convert_onnx_to_om.sh 转换为昇腾 .om 文件。

用法：
  # 全部导出（batch=1,4,8,16；Qwen3 仅 1,4,8 以节省内存）
  python export_bert_gpt2_qwen.py

  # 仅导出特定模型
  python export_bert_gpt2_qwen.py --model bert_base
  python export_bert_gpt2_qwen.py --model gpt2_small
  python export_bert_gpt2_qwen.py --model qwen3_06b

  # 自定义输出目录和 batch 列表
  python export_bert_gpt2_qwen.py --output_dir /tmp/onnx --batch 1 4

依赖：
  pip install torch transformers onnx
  # Qwen3-0.6B 还需要：
  pip install transformers>=4.51.0

模型规格（与 roofline_910b4_calibrated.py CANDIDATE_MODELS 一致）：
  BERT-base：L=12, d=768, S=128；输入 input_ids+attention_mask+token_type_ids
  GPT-2-small：L=12, d=768, S=512；输入 input_ids（use_cache=False）
  Qwen3-0.6B：L=28, d=1024, S=512；GQA 16Q/8KV，SwiGLU；输入 input_ids（use_cache=False）

下一步：
  bash convert_onnx_to_om.sh <onnx_dir> <om_dir>
"""

import argparse
import sys
from pathlib import Path

import torch


OUTPUT_DIR = Path(__file__).parent.parent / "models"   # sim-experiment/models/


# ─────────────────────────────────────────────────────────────
# 1. BERT-base（google-bert/bert-base-uncased）
#    L=12, d=768, S=128；三输入：input_ids / attention_mask / token_type_ids
# ─────────────────────────────────────────────────────────────
def export_bert_base(batches, output_dir: Path):
    print("\n[1/3] BERT-base (google-bert/bert-base-uncased)")
    try:
        from transformers import BertModel
    except ImportError:
        print("  ✗ 需要 transformers：pip install transformers")
        return

    print("  正在从 HuggingFace 下载 BERT-base（约 440 MB）…")
    model = BertModel.from_pretrained("google-bert/bert-base-uncased")
    model.eval()
    model.float()   # FP32 用于 ONNX 导出；atc 在编译时量化到 FP16

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量：{total_params:,}  ({total_params / 1e6:.1f}M，约 {total_params*2/1e6:.0f} MB FP16)")

    for B in batches:
        S = 128
        dummy_ids   = torch.zeros(B, S, dtype=torch.long)
        dummy_mask  = torch.ones(B, S, dtype=torch.long)
        dummy_types = torch.zeros(B, S, dtype=torch.long)

        out_path = output_dir / f"bert_base_batch_{B}.onnx"
        print(f"  导出 batch={B} → {out_path.name} …")

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_ids, dummy_mask, dummy_types),
                str(out_path),
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["last_hidden_state", "pooler_output"],
                dynamic_axes={
                    "input_ids":        {0: "batch"},
                    "attention_mask":   {0: "batch"},
                    "token_type_ids":   {0: "batch"},
                    "last_hidden_state":{0: "batch"},
                    "pooler_output":    {0: "batch"},
                },
                opset_version=14,
                do_constant_folding=True,
            )
        print(f"  ✓ batch={B}  size={out_path.stat().st_size / 1e6:.1f} MB")


# ─────────────────────────────────────────────────────────────
# 2. GPT-2-small（openai-community/gpt2）
#    L=12, d=768, S=512；单输入 input_ids；use_cache=False
# ─────────────────────────────────────────────────────────────
class GPT2Wrapper(torch.nn.Module):
    """
    包装 GPT2LMHeadModel，强制 use_cache=False 并只返回 logits，
    避免 ONNX tracer 对 past_key_values（动态结构）的报错。
    """
    def __init__(self, gpt2_model):
        super().__init__()
        self.model = gpt2_model

    def forward(self, input_ids):
        out = self.model(input_ids=input_ids, use_cache=False)
        return out.logits


def _patch_sdpa():
    """
    Monkey-patch torch.nn.functional.scaled_dot_product_attention 为纯矩阵乘法。
    PyTorch 2.8 的 SDPA 内部走 vmap/functorch dispatch，与 TorchScript tracer 不兼容；
    替换成标准 QK^TV 实现可绕过该 bug。
    """
    import math
    import torch.nn.functional as F

    def _manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, **kwargs):
        head_dim = query.size(-1)
        s = (scale if scale is not None else 1.0 / math.sqrt(head_dim))
        scores = torch.matmul(query, key.transpose(-2, -1)) * s
        if is_causal:
            L, S = query.size(-2), key.size(-2)
            mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        return torch.matmul(weights, value)

    F.scaled_dot_product_attention = _manual_sdpa
    torch.nn.functional.scaled_dot_product_attention = _manual_sdpa


def export_gpt2_small(batches, output_dir: Path):
    print("\n[2/3] GPT-2-small (openai-community/gpt2)")
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("  ✗ 需要 transformers：pip install transformers")
        return

    # Patch SDPA before loading model（ONNX tracer 兼容性修复）
    _patch_sdpa()

    print("  正在从 HuggingFace 下载 GPT-2-small（约 548 MB）…")
    base_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    base_model.eval()
    base_model.float()

    model = GPT2Wrapper(base_model)

    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  参数量：{total_params:,}  ({total_params / 1e6:.1f}M，约 {total_params*2/1e6:.0f} MB FP16)")

    for B in batches:
        S = 512
        dummy_ids = torch.zeros(B, S, dtype=torch.long)

        out_path = output_dir / f"gpt2_small_batch_{B}.onnx"
        print(f"  导出 batch={B} → {out_path.name} …")

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_ids,
                str(out_path),
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch"},
                    "logits":    {0: "batch"},
                },
                opset_version=14,
                do_constant_folding=True,
            )
        print(f"  ✓ batch={B}  size={out_path.stat().st_size / 1e6:.1f} MB")


# ─────────────────────────────────────────────────────────────
# 3. Qwen3-0.6B（Qwen/Qwen3-0.6B）
#    L=28, d=1024, GQA 16Q/8KV, SwiGLU, S=512；单输入 input_ids；use_cache=False
#    attn_implementation="eager" 禁用 Flash Attention，保证 ONNX 可导出
# ─────────────────────────────────────────────────────────────
def _patch_qwen3_masking():
    """
    修复 transformers 5.x masking_utils.sdpa_mask BC 代码在 ONNX TorchScript
    tracing 时崩溃的 bug。

    根因：masking_utils.sdpa_mask 第 492 行有向后兼容代码：
        if isinstance(q_length, torch.Tensor):
            q_length, q_offset = q_length.shape[0], q_length[0].to(device)
    当 q_length 是 0-dim tensor（ONNX 追踪过程中 inputs_embeds.shape[1]
    或 torch.tensor(seq_len) 等路径生成的标量张量）时，shape=() 导致
    shape[0] 抛出 IndexError: tuple index out of range。

    修复策略：在 mu.sdpa_mask 外层包装一个安全版本，把 0-dim tensor 实参
    转换为 Python int，再交给原函数处理。eager_mask 在同一 module 内通过
    全局命名空间引用 sdpa_mask，覆写 mu.sdpa_mask 即可生效，无需修改库文件。
    """
    try:
        import transformers.masking_utils as mu
        if not hasattr(mu, "sdpa_mask"):
            return  # transformers < 5.x 没有此函数，无需修复

        _orig = mu.sdpa_mask

        def _safe_sdpa_mask(*args, **kwargs):
            # 修复位置参数：signature (batch_size, q_length, kv_length, ...)
            args = list(args)
            for idx in (1, 2):
                if (idx < len(args)
                        and isinstance(args[idx], torch.Tensor)
                        and args[idx].dim() == 0):
                    args[idx] = int(args[idx].item())
            # 修复关键字参数
            for key in ("q_length", "kv_length", "q_offset"):
                if (key in kwargs
                        and isinstance(kwargs[key], torch.Tensor)
                        and kwargs[key].dim() == 0):
                    kwargs[key] = int(kwargs[key].item())
            return _orig(*args, **kwargs)

        mu.sdpa_mask = _safe_sdpa_mask
        print("  [patch] masking_utils.sdpa_mask: 0-dim tensor BC 修复已应用")
    except Exception as exc:
        print(f"  [patch] masking_utils 补丁跳过（版本不适用）: {exc}")


def _patch_torch_diff_for_onnx():
    """
    替换 torch.diff 为 ONNX 可导出的 narrow+sub 实现。

    根因：aten::diff 在 opset 14（乃至任何已注册的 opset）均无 ONNX symbolic，
    PyTorch ONNX 导出器在 _jit_pass_onnx 阶段报
    UnsupportedOperatorError: Exporting the operator 'aten::diff' is not supported.

    修复策略：在 torch.onnx.export 调用前，将 torch.diff 替换为等价的
    torch.narrow（→ ONNX Slice）+ 减法（→ ONNX Sub）实现，使 tracer
    只记录 ONNX 支持的算子。

    注意：此实现在 tracing 时把序列长度固化为常量（TracerWarning），
    但因为我们 dynamic_axes 只标记 batch 维度为动态，序列长度本身已静态，
    所以对 Qwen3 ONNX 导出场景完全安全。
    """
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
            seq = result.shape[dim]          # tracing 时为具体 int，安全
            lo = torch.narrow(result, dim, 0, seq - 1)   # → ONNX Slice
            hi = torch.narrow(result, dim, 1, seq - 1)   # → ONNX Slice
            result = hi - lo                              # → ONNX Sub
        return result

    torch.diff = _onnx_safe_diff
    print("  [patch] torch.diff: ONNX Slice+Sub 替代实现已应用")


class CausalLMWrapper(torch.nn.Module):
    """
    包装 AutoModelForCausalLM，强制 use_cache=False 并只返回 logits。
    适用于 Qwen3 / LLaMA 等 HF causal LM 系列。
    """
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids):
        out = self.model(input_ids=input_ids, use_cache=False)
        return out.logits


def export_qwen3_06b(batches, output_dir: Path):
    print("\n[3/3] Qwen3-0.6B (Qwen/Qwen3-0.6B)")
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("  ✗ 需要 transformers >= 4.51.0：pip install -U transformers")
        return

    # 修复 transformers 5.x masking_utils BC 代码崩溃（0-dim tensor IndexError）
    _patch_qwen3_masking()
    # 修复 aten::diff 无 ONNX symbolic 的问题（用 Slice+Sub 替代）
    _patch_torch_diff_for_onnx()

    print("  正在从 HuggingFace 下载 Qwen3-0.6B（约 1.2 GB）…")
    print("  ⚠ 首次下载较慢，后续使用本地缓存")

    # attn_implementation="eager" 避免 SDPA/Flash Attention 导出问题
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        attn_implementation="eager",
        torch_dtype=torch.float32,   # ONNX 导出用 FP32；atc 编译时再量化 FP16
    )
    base_model.eval()

    model = CausalLMWrapper(base_model)

    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  参数量：{total_params:,}  ({total_params / 1e6:.1f}M，约 {total_params*2/1e6:.0f} MB FP16)")

    for B in batches:
        S = 512
        dummy_ids = torch.zeros(B, S, dtype=torch.long)

        out_path = output_dir / f"qwen3_06b_batch_{B}.onnx"
        print(f"  导出 batch={B} → {out_path.name} …")
        print("  （Qwen3 ONNX 导出较慢，约 1-2 分钟，请耐心等待）")

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_ids,
                str(out_path),
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch"},
                    "logits":    {0: "batch"},
                },
                opset_version=14,
                do_constant_folding=True,
            )
        print(f"  ✓ batch={B}  size={out_path.stat().st_size / 1e6:.1f} MB")


# ─────────────────────────────────────────────────────────────
# 验证 ONNX 文件完整性
# ─────────────────────────────────────────────────────────────
def verify_onnx(output_dir: Path, prefixes):
    print("\n─── 验证 ONNX 文件 ───")
    try:
        import onnx
        import numpy as np
    except ImportError:
        print("  ✗ 需要 onnx：pip install onnx")
        return

    any_found = False
    for onnx_file in sorted(output_dir.glob("*.onnx")):
        if not any(onnx_file.name.startswith(p) for p in prefixes):
            continue
        try:
            m = onnx.load(str(onnx_file))
            onnx.checker.check_model(m)
            weight_bytes = sum(
                np.prod(list(t.dims)) * 4   # FP32 = 4 bytes
                for t in m.graph.initializer
            )
            inputs = [(i.name, [d.dim_value or "?"
                                for d in i.type.tensor_type.shape.dim])
                      for i in m.graph.input]
            print(f"  ✓ {onnx_file.name}: {weight_bytes/1e6:.1f} MB FP32, inputs={inputs}")
            any_found = True
        except Exception as e:
            print(f"  ✗ {onnx_file.name}: {e}")

    if not any_found:
        print("  （未找到匹配的 ONNX 文件）")


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="下载并导出 BERT-base / GPT-2-small / Qwen3-0.6B ONNX"
    )
    parser.add_argument(
        "--model",
        choices=["bert_base", "gpt2_small", "qwen3_06b", "all"],
        default="all",
        help="要导出的模型（默认 all）",
    )
    parser.add_argument(
        "--batch", nargs="+", type=int, default=None,
        help="batch size 列表（默认：bert/gpt2=1,4,8,16；qwen3=1,4,8）",
    )
    parser.add_argument(
        "--output_dir", default=str(OUTPUT_DIR),
        help=f"ONNX 输出目录（默认 {OUTPUT_DIR}）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 验证 PyTorch
    print(f"PyTorch {torch.__version__}")
    print(f"输出目录：{output_dir.resolve()}")

    # 各模型默认 batch 列表（Qwen3 大模型，限 B≤8 避免显存/内存不足）
    default_batches = {
        "bert_base":   [1, 4, 8, 16],
        "gpt2_small":  [1, 4, 8, 16],
        "qwen3_06b":   [1, 4, 8],
    }

    to_export = (
        ["bert_base", "gpt2_small", "qwen3_06b"]
        if args.model == "all"
        else [args.model]
    )

    for model_name in to_export:
        batches = args.batch if args.batch else default_batches[model_name]
        if model_name == "bert_base":
            export_bert_base(batches, output_dir)
        elif model_name == "gpt2_small":
            export_gpt2_small(batches, output_dir)
        elif model_name == "qwen3_06b":
            export_qwen3_06b(batches, output_dir)

    # 验证
    verify_onnx(output_dir, prefixes=["bert_base", "gpt2_small", "qwen3_06b"])

    print("\n完成。下一步：")
    print(f"  bash benchmark/convert_onnx_to_om.sh {output_dir} <om_output_dir>")


if __name__ == "__main__":
    main()
