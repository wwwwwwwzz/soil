"""
高光谱Transformer模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class SpectralSpatialTransformer(nn.Module):
    def __init__(
        self,
        num_bands=512,
        patch_size=10,
        d_model=256,
        n_heads=8,
        n_layers=6,
        num_classes=5,
        dropout=0.1
    ):
        super().__init__()

        self.num_bands = num_bands
        self.patch_size = patch_size
        self.d_model = d_model

        # 光谱嵌入
        self.spectral_embedding = nn.Linear(num_bands, d_model)

        # 位置编码（用于10x10的空间结构）
        self.pos_embedding = nn.Parameter(torch.randn(1, patch_size * patch_size, d_model))

        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 分类和回归头
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.regressor = nn.Linear(d_model, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, height=10, width=10, bands=512)
        B, H, W, C = x.shape

        # 重塑为序列：(batch, h*w, bands)
        x = rearrange(x, 'b h w c -> b (h w) c')

        # 光谱嵌入
        x = self.spectral_embedding(x)  # (B, 100, d_model)

        # 添加位置编码
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.dropout(x)

        # 通过Transformer层
        for layer in self.transformer_layers:
            x = layer(x)

        # 全局池化
        x = x.mean(dim=1)  # (B, d_model)
        x = self.norm(x)

        # 双任务输出
        class_logits = self.classifier(x)
        regression = self.regressor(x)

        return class_logits, regression

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        # 多头注意力
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 自注意力
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # FFN
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)

        return x