"""
固定网络 AI 场景模型 ONNX 导出脚本
导出 ET-BERT、NetGPT、MalConv2、Kitsune 的 ONNX 文件
运行：python3 export_models_onnx.py
"""

import os
import sys
import json
import struct
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. ET-BERT（AI-DPI，BERT-base 变体，vocab=256）
# ─────────────────────────────────────────────────────────────
def export_et_bert():
    print("\n[1/4] ET-BERT (AI-DPI)...")
    try:
        import torch
        from transformers import AutoConfig, AutoModel
    except ImportError:
        print("  ✗ 需要 torch 和 transformers：pip3 install torch transformers")
        return

    # ET-BERT 使用 BERT-base 架构，vocab=256（字节级 token）
    # 来源：github.com/linwhitehat/ET-BERT，bert_base_config.json
    config_dict = {
        "architectures": ["BertModel"],
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 256,
    }

    from transformers import BertConfig, BertModel
    config = BertConfig(**config_dict)
    model = BertModel(config)
    model.eval()

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  INT8 大小: {total_params/1e6:.2f} MB")

    # 导出 ONNX（seq_len=128，batch=1）
    seq_len = 128
    dummy_input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    out_path = OUTPUT_DIR / "et_bert_batch_1.onnx"
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  ✓ 导出: {out_path}")


# ─────────────────────────────────────────────────────────────
# 2. NetGPT（流量大模型，GPT-2 base 变体，vocab=256）
# ─────────────────────────────────────────────────────────────
def export_netgpt():
    print("\n[2/4] NetGPT (流量大模型)...")
    print("  ⚠ PyTorch 2.8 TorchScript tracer 对 GPT-2 有已知 bug（unordered_map::at）")
    print("  → 跳过 ONNX 导出，使用 netgpt.yaml 进行 Roofline 分析（架构参数已从 GitHub 确认）")
    print("  参数量: 86,039,040 (86.04M)，INT8 大小: 86.04 MB")



# ─────────────────────────────────────────────────────────────
# 3. MalConv2（文件检测，1D CNN + Global Channel Gating）
# ─────────────────────────────────────────────────────────────
def export_malconv2():
    print("\n[3/4] MalConv2 (文件检测)...")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  ✗ 需要 torch")
        return

    # MalConv2 架构（来自 github.com/NeuromorphicComputationResearchProgram/MalConv2）
    # channels=256, window_size=256, stride=64, embd_size=8
    class MalConv2(nn.Module):
        def __init__(self, channels=256, window_size=256, stride=64, embd_size=8):
            super().__init__()
            self.embd = nn.Embedding(257, embd_size, padding_idx=0)
            # Gated conv: 两路并行（value + gate）
            self.conv_1 = nn.Conv1d(embd_size, channels, window_size, stride=stride, bias=True)
            self.conv_2 = nn.Conv1d(embd_size, channels, window_size, stride=stride, bias=True)
            # Global Channel Gating
            self.fc_1 = nn.Linear(channels, channels)
            self.fc_2 = nn.Linear(channels, channels)
            self.fc_out = nn.Linear(channels, 2)

        def forward(self, x):
            # x: [batch, seq_len] int64
            x = self.embd(x)                          # [B, L, 8]
            x = x.permute(0, 2, 1)                    # [B, 8, L]
            cnn_value = self.conv_1(x)                # [B, 256, T]
            cnn_gate = torch.sigmoid(self.conv_2(x))  # [B, 256, T]
            x = cnn_value * cnn_gate                  # gated
            x = x.max(dim=-1).values                  # global max pool [B, 256]
            # Global Channel Gating
            gate = torch.sigmoid(self.fc_1(x))
            x = self.fc_2(x) * gate
            return self.fc_out(x)

    model = MalConv2()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.3f}M)")
    print(f"  INT8 大小: {total_params/1e6:.3f} MB")

    # 输入：文件字节序列，截断到 1MB（1,048,576 bytes）
    seq_len = 1_048_576
    dummy_input = torch.zeros(1, seq_len, dtype=torch.long)

    out_path = OUTPUT_DIR / "malconv2_batch_1.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["file_bytes"],
        output_names=["logits"],
        dynamic_axes={
            "file_bytes": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  ✓ 导出: {out_path}")


# ─────────────────────────────────────────────────────────────
# 4. Kitsune（DDoS/SmartAD，KitNET 集成自编码器）
# ─────────────────────────────────────────────────────────────
def export_kitsune():
    print("\n[4/4] Kitsune (DDoS/SmartAD)...")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  ✗ 需要 torch")
        return

    # Kitsune/KitNET 架构（来自 github.com/ymirsky/Kitsune-py，NDSS'18）
    # 集成自编码器：maxAE=10 个小 AE + 1 个输出 AE
    # 特征维度：100（AfterImage 统计特征，典型值）
    # 隐层比例：0.75（hidden = ceil(input * 0.75)）
    class SingleAE(nn.Module):
        def __init__(self, input_dim, hidden_ratio=0.75):
            super().__init__()
            hidden = max(1, int(np.ceil(input_dim * hidden_ratio)))
            self.encoder = nn.Linear(input_dim, hidden)
            self.decoder = nn.Linear(hidden, input_dim)

        def forward(self, x):
            h = torch.relu(self.encoder(x))
            return self.decoder(h)

    class KitNET(nn.Module):
        """
        KitNET: 集成自编码器
        - n_features 个输入特征分配给 maxAE 个子 AE
        - 每个子 AE 的重建误差作为输出 AE 的输入
        """
        def __init__(self, n_features=100, maxAE=10, hidden_ratio=0.75):
            super().__init__()
            # 每个子 AE 处理 n_features/maxAE 个特征
            features_per_ae = int(np.ceil(n_features / maxAE))
            self.ensemble = nn.ModuleList([
                SingleAE(features_per_ae, hidden_ratio) for _ in range(maxAE)
            ])
            # 输出 AE：输入是各子 AE 的重建误差（maxAE 维）
            self.output_ae = SingleAE(maxAE, hidden_ratio)
            self.n_features = n_features
            self.maxAE = maxAE
            self.features_per_ae = features_per_ae

        def forward(self, x):
            # x: [batch, n_features]
            rmse_list = []
            for i, ae in enumerate(self.ensemble):
                start = i * self.features_per_ae
                end = min(start + self.features_per_ae, self.n_features)
                x_i = x[:, start:end]
                # 补零到 features_per_ae
                if x_i.shape[1] < self.features_per_ae:
                    pad = torch.zeros(x_i.shape[0], self.features_per_ae - x_i.shape[1])
                    x_i = torch.cat([x_i, pad], dim=1)
                recon = ae(x_i)
                rmse = torch.sqrt(torch.mean((x_i - recon) ** 2, dim=1, keepdim=True))
                rmse_list.append(rmse)
            rmse_vec = torch.cat(rmse_list, dim=1)  # [batch, maxAE]
            anomaly_score = self.output_ae(rmse_vec)
            return anomaly_score

    model = KitNET(n_features=100, maxAE=10)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} (~{total_params/1000:.1f}K)")
    print(f"  INT8 大小: ~{total_params/1000:.1f} KB")

    dummy_input = torch.zeros(1, 100, dtype=torch.float32)

    out_path = OUTPUT_DIR / "kitsune_batch_1.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["features"],
        output_names=["anomaly_score"],
        dynamic_axes={
            "features": {0: "batch"},
            "anomaly_score": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  ✓ 导出: {out_path}")


# ─────────────────────────────────────────────────────────────
# 验证导出的 ONNX 文件
# ─────────────────────────────────────────────────────────────
def verify_exports():
    print("\n─── 验证导出文件 ───")
    try:
        import onnx
        import numpy as np
    except ImportError:
        print("  ✗ 需要 onnx：pip3 install onnx")
        return

    for onnx_file in sorted(OUTPUT_DIR.glob("*.onnx")):
        try:
            m = onnx.load(str(onnx_file))
            onnx.checker.check_model(m)
            total = sum(np.prod(list(t.dims)) for t in m.graph.initializer)
            inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                      for i in m.graph.input]
            print(f"  ✓ {onnx_file.name}: {total:,} params, inputs={inputs}")
        except Exception as e:
            print(f"  ✗ {onnx_file.name}: {e}")


if __name__ == "__main__":
    print("固定网络 AI 场景模型 ONNX 导出")
    print(f"输出目录: {OUTPUT_DIR}")

    # 检查 torch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except ImportError:
        print("✗ 未安装 PyTorch，请先运行：pip3 install torch")
        sys.exit(1)

    export_et_bert()
    export_netgpt()
    export_malconv2()
    export_kitsune()
    verify_exports()

    print("\n完成。")
