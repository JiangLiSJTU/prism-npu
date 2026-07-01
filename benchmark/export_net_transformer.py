"""
Net-Transformer 等效 PyTorch 实现
用于导出 ONNX，再通过 atc 转换为昇腾 .om 文件

结构来源：NetTrans_structure_template.md（人工核对，2026-04-28）
参数：S=256, D_model=384, H=3(GQA, g=3), D_head=128, D_ffn=1536, L=1
      CLS_dim=1152, D_out=1024, 激活函数=ReLU, Pre-norm=True

注意：
- 权重为随机初始化，仅用于测量推理延迟（硬件瓶颈分析），不代表真实模型精度
- GQA: Q有3头，K/V各1组（g=3，每组1头），D_head=128
- 分类头取 CLS token（序列第0位）
"""

import argparse
import json
import math
import torch
import torch.nn as nn
from pathlib import Path


class NetTransformerGQA(nn.Module):
    """
    Net-Transformer 等效结构
    Group-Query Attention + FFN(ReLU) + Pre-LayerNorm + 分类头
    """

    def __init__(
        self,
        seq_len: int = 256,
        d_in: int = 384,
        d_model: int = 384,
        num_heads: int = 3,       # Q heads
        num_kv_groups: int = 3,   # g=3，每组1个KV头，共3个KV头（等同于MHA）
        d_head: int = 128,
        d_ffn: int = 1536,
        num_layers: int = 1,
        cls_dim: int = 1152,
        d_out: int = 1024,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.d_head = d_head
        self.num_layers = num_layers

        # 输入嵌入（4个 Gather + 1个 Add，用线性层近似）
        self.input_embed = nn.Linear(d_in, d_model, bias=False)

        # Encoder 层
        self.layers = nn.ModuleList([
            NetTransEncoderLayer(d_model, num_heads, num_kv_groups, d_head, d_ffn)
            for _ in range(num_layers)
        ])

        # 分类头（取 CLS token，即序列第0位）
        # CLS Token投影 × 2 + 输出投影
        self.cls_proj1 = nn.Linear(d_model, cls_dim, bias=False)
        self.cls_proj2 = nn.Linear(cls_dim, cls_dim, bias=False)
        self.cls_out   = nn.Linear(cls_dim, d_out, bias=False)
        self.cls_relu  = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D_in]
        x = self.input_embed(x)          # [B, S, D_model]

        for layer in self.layers:
            x = layer(x)                 # [B, S, D_model]

        # 取 CLS token（第0位）
        cls = x[:, 0, :]                 # [B, D_model]
        cls = self.cls_relu(self.cls_proj1(cls))   # [B, CLS_dim]
        cls = self.cls_relu(self.cls_proj2(cls))   # [B, CLS_dim]
        out = self.cls_out(cls)                    # [B, D_out]
        return out


class NetTransEncoderLayer(nn.Module):
    """单个 Encoder 层：Pre-LN + GQA + Pre-LN + FFN(ReLU)"""

    def __init__(self, d_model, num_heads, num_kv_groups, d_head, d_ffn):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.d_head = d_head

        # Pre-norm（Attention 前）
        self.norm1 = nn.LayerNorm(d_model)

        # GQA 投影
        self.q_proj = nn.Linear(d_model, num_heads * d_head, bias=False)
        # K/V: num_kv_groups 个头（g=3时等同于MHA，每组1头）
        kv_heads = num_heads // num_kv_groups  # = 1
        self.k_proj = nn.Linear(d_model, kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * d_head, bias=False)
        self.out_proj = nn.Linear(num_heads * d_head, d_model, bias=False)

        # Pre-norm（FFN 前）
        self.norm2 = nn.LayerNorm(d_model)

        # FFN with ReLU
        self.ffn_linear1 = nn.Linear(d_model, d_ffn, bias=False)
        self.ffn_relu    = nn.ReLU()
        self.ffn_linear2 = nn.Linear(d_ffn, d_model, bias=False)

        self.scale = math.sqrt(d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # ── Attention ──────────────────────────────────────────
        residual = x
        x = self.norm1(x)

        Q = self.q_proj(x)  # [B, S, num_heads*d_head]
        K = self.k_proj(x)  # [B, S, kv_heads*d_head]
        V = self.v_proj(x)  # [B, S, kv_heads*d_head]

        # Reshape for multi-head attention
        Q = Q.view(B, S, self.num_heads, self.d_head).transpose(1, 2)  # [B, H, S, d_head]
        kv_heads = self.num_heads // self.num_kv_groups
        K = K.view(B, S, kv_heads, self.d_head).transpose(1, 2)        # [B, kv_H, S, d_head]
        V = V.view(B, S, kv_heads, self.d_head).transpose(1, 2)        # [B, kv_H, S, d_head]

        # GQA: 扩展 K/V 以匹配 Q 的头数
        if kv_heads < self.num_heads:
            repeat = self.num_heads // kv_heads
            K = K.repeat_interleave(repeat, dim=1)  # [B, H, S, d_head]
            V = V.repeat_interleave(repeat, dim=1)  # [B, H, S, d_head]

        # QK^T + ReLU（模板中用 ReLU 替代 Softmax）
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, S, S]
        attn = torch.relu(attn)                                     # ReLU（非 Softmax）

        # AV
        out = torch.matmul(attn, V)                                 # [B, H, S, d_head]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)       # [B, S, H*d_head]
        out = self.out_proj(out)                                     # [B, S, D_model]
        x = residual + out

        # ── FFN ────────────────────────────────────────────────
        residual = x
        x = self.norm2(x)
        x = self.ffn_relu(self.ffn_linear1(x))  # [B, S, D_ffn]
        x = self.ffn_linear2(x)                  # [B, S, D_model]
        x = residual + x

        return x


def export_onnx(batch: int, output_dir: Path):
    """导出指定 batch 的 ONNX 文件"""
    model = NetTransformerGQA()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.3f}M)")

    # 输入：[B, S, D_in]，D_in=384
    dummy_input = torch.zeros(batch, 256, 384, dtype=torch.float32)

    out_path = output_dir / f"net_transformer_batch_{batch}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["input_features"],
        output_names=["logits"],
        dynamic_axes={
            "input_features": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  ✓ 导出: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Net-Transformer ONNX 导出")
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 4, 8, 16],
                        help="要导出的 batch size 列表")
    parser.add_argument("--output_dir", default=".",
                        help="ONNX 输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Net-Transformer ONNX 导出")
    print(f"结构：S=256, D_model=384, H=3(GQA), D_head=128, D_ffn=1536, L=1")
    print(f"分类头：CLS_dim=1152, D_out=1024, 激活=ReLU, Pre-norm=True")
    print(f"输出目录：{output_dir}")

    for batch in args.batch:
        print(f"\nbatch={batch}...")
        export_onnx(batch, output_dir)

    print("\n完成。")
    print("下一步：用 atc 转换为 .om 文件（见 convert_onnx_to_om.sh）")


if __name__ == "__main__":
    main()
