"""
physics_constrained_model.py
物理约束的Spectral-Spatial Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt


class PhysicsConstrainedTransformer(nn.Module):
    def __init__(
            self,
            num_bands=512,
            wavelength_start=889,
            wavelength_end=1710,
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_classes=3,
            dropout=0.3
    ):
        super().__init__()

        self.num_bands = num_bands
        self.wavelengths = torch.linspace(wavelength_start, wavelength_end, num_bands)

        # 原始Transformer组件
        self.spectral_embedding = nn.Linear(num_bands, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model))

        # 物理引导的注意力模块
        self.physics_attention = PhysicsGuidedAttention(
            num_bands=num_bands,
            wavelengths=self.wavelengths,
            d_model=d_model
        )

        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 输出头
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.regressor = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        B, H, W, C = x.shape

        # 提取物理特征
        physics_features, absorption_attention = self.physics_attention(x)

        # 标准Transformer处理
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.spectral_embedding(x)
        x = x + self.pos_embedding[:, :x.size(1), :]

        # 融合物理特征
        x = x + physics_features.unsqueeze(1).expand(-1, x.size(1), -1) * 0.1
        x = self.dropout(x)

        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)

        # 池化和输出
        x = x.mean(dim=1)
        x = self.norm(x)

        class_logits = self.classifier(x)
        regression = self.regressor(x)

        if return_attention:
            return class_logits, regression, absorption_attention
        return class_logits, regression


"""
physics_constrained_model.py
物理约束的Spectral-Spatial Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt


class PhysicsConstrainedTransformer(nn.Module):
    def __init__(
            self,
            num_bands=512,
            wavelength_start=889,
            wavelength_end=1710,
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_classes=3,
            dropout=0.3
    ):
        super().__init__()

        self.num_bands = num_bands
        self.wavelengths = torch.linspace(wavelength_start, wavelength_end, num_bands)

        # 原始Transformer组件
        self.spectral_embedding = nn.Linear(num_bands, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model))

        # 物理引导的注意力模块
        self.physics_attention = PhysicsGuidedAttention(
            num_bands=num_bands,
            wavelengths=self.wavelengths,
            d_model=d_model
        )

        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 输出头
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.regressor = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        B, H, W, C = x.shape

        # 提取物理特征
        physics_features, absorption_attention = self.physics_attention(x)

        # 标准Transformer处理
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.spectral_embedding(x)
        x = x + self.pos_embedding[:, :x.size(1), :]

        # 融合物理特征
        x = x + physics_features.unsqueeze(1).expand(-1, x.size(1), -1) * 0.1
        x = self.dropout(x)

        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)

        # 池化和输出
        x = x.mean(dim=1)
        x = self.norm(x)

        class_logits = self.classifier(x)
        regression = self.regressor(x)

        if return_attention:
            return class_logits, regression, absorption_attention
        return class_logits, regression


class PhysicsGuidedAttention(nn.Module):
    """物理引导的注意力机制"""

    def __init__(self, num_bands, wavelengths, d_model):
        super().__init__()
        self.num_bands = num_bands
        self.wavelengths = wavelengths
        self.d_model = d_model

        # 构建碳氢化合物标准吸收曲线并注册为buffer
        absorption_curve = self.create_hydrocarbon_absorption_curve()
        self.register_buffer('absorption_curve', absorption_curve)

        # 学习权重参数
        self.absorption_weight = nn.Parameter(torch.ones(num_bands))
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_bands, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def create_hydrocarbon_absorption_curve(self):
        """创建碳氢化合物的标准吸收曲线"""
        wavelengths = self.wavelengths.numpy()
        absorption = np.zeros(len(wavelengths))

        # 基于文献的主要吸收峰
        absorption_peaks = {
            930: 0.3,   # C-H stretch (第三泛音)
            1020: 0.2,  # 芳烃C-H
            1195: 0.4,  # CH2组合带
            1215: 0.5,  # C-H组合带
            1380: 0.3,  # CH3变形
            1400: 0.6,  # O-H (水分干扰)
            1417: 0.4,  # 芳烃
            1680: 0.7,  # 碳氢化合物第一泛音
            1695: 0.6,  # CH2第一泛音
            1705: 0.8,  # CH第一泛音
            1725: 0.9,  # 主吸收区起始
            1730: 1.0,  # 石油主吸收峰
            1760: 0.95, # 长链烷烃
            1765: 0.9,  # 芳烃
            1780: 0.85, # 环烷烃
            1790: 0.8,  # 支链烷烃
            1815: 0.6,  # 尾部吸收
        }

        # 使用高斯函数模拟吸收峰
        for center_wl, intensity in absorption_peaks.items():
            if 889 <= center_wl <= 1710:
                sigma = 10  # 带宽
                idx = np.argmin(np.abs(wavelengths - center_wl))
                gaussian = intensity * np.exp(-0.5 * ((wavelengths - center_wl) / sigma) ** 2)
                absorption += gaussian

        # 归一化
        absorption = absorption / absorption.max()

        # 添加基线
        baseline = 0.1 + 0.05 * np.sin(2 * np.pi * (wavelengths - 889) / (1710 - 889))
        absorption = 0.8 * absorption + 0.2 * baseline

        return torch.tensor(absorption, dtype=torch.float32)

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape

        # 计算平均光谱
        mean_spectrum = x.mean(dim=(1, 2))  # (B, C)

        # 计算与标准吸收曲线的相似度
        # absorption_curve现在作为buffer会自动在正确设备上
        absorption_attention = F.cosine_similarity(
            mean_spectrum,
            self.absorption_curve.unsqueeze(0).expand(B, -1),
            dim=-1
        ).unsqueeze(-1)  # (B, 1)

        # 加权光谱特征
        weighted_spectrum = mean_spectrum * self.absorption_weight

        # 提取物理特征
        physics_features = self.feature_extractor(weighted_spectrum)

        return physics_features, absorption_attention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        return x