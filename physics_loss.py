"""
physics_loss.py
物理约束损失函数 - 修正版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhysicsConstrainedLoss(nn.Module):
    """包含物理约束的多任务损失"""

    def __init__(self, alpha=0.45, beta=0.45, gamma=0.10):
        super().__init__()
        self.alpha = alpha  # 分类权重
        self.beta = beta   # 回归权重
        self.gamma = gamma  # 物理约束权重

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def calculate_peak_alignment_loss(self, absorption_weights, reference_curve):
        """计算峰值对齐损失"""
        device = absorption_weights.device

        # 确保在同一设备上
        if reference_curve.device != device:
            reference_curve = reference_curve.to(device)

        # 创建峰值掩码（关注主要吸收峰区域）
        peak_mask = torch.zeros_like(reference_curve)

        # 根据波长范围计算索引
        # 889-1710nm映射到0-512波段
        peak_wavelengths = [1680, 1695, 1705, 1725, 1730, 1760, 1765, 1780, 1790]

        for wl in peak_wavelengths:
            idx = int((wl - 889) / (1710 - 889) * 512)
            # 在峰值周围创建高斯权重
            for i in range(max(0, idx-10), min(len(peak_mask), idx+11)):
                # 修正：创建torch tensor而不是float
                weight = torch.exp(torch.tensor(-0.5 * ((i - idx) / 5.0) ** 2, device=device))
                peak_mask[i] = torch.max(peak_mask[i], weight)

        # 计算加权MSE损失，更关注峰值区域
        weighted_loss = F.mse_loss(
            absorption_weights * peak_mask,
            reference_curve * peak_mask,
            reduction='mean'
        )

        return weighted_loss

    def forward(self, outputs, targets, model):
        """
        outputs: (class_logits, regression, absorption_attention)
        targets: (labels, normalized_concentrations, original_concentrations)
        """
        class_logits, regression, absorption_attention = outputs
        labels, norm_concentrations, orig_concentrations = targets

        # 获取设备
        device = class_logits.device

        # 1. 分类损失
        loss_cls = self.ce_loss(class_logits, labels)

        # 2. 回归损失
        loss_reg = self.mse_loss(regression.squeeze(), norm_concentrations)

        # 3. 物理约束损失
        # 3.1 光谱注意力应该关注已知吸收峰
        absorption_weights = model.physics_attention.absorption_weight
        reference_curve = model.physics_attention.absorption_curve

        # 确保在同一设备上
        if reference_curve.device != device:
            reference_curve = reference_curve.to(device)

        # 正则化：学习的权重应该与参考曲线相关
        physics_loss_spectral = 1 - F.cosine_similarity(
            absorption_weights.unsqueeze(0),
            reference_curve.unsqueeze(0),
            dim=1
        ).mean()

        # 3.2 峰值对齐损失
        physics_loss_peaks = self.calculate_peak_alignment_loss(
            absorption_weights,
            reference_curve
        )

        # 3.3 浓度预测应该与吸收强度相关（Beer-Lambert定律）
        log_concentrations = torch.log1p(orig_concentrations).to(device)
        predicted_absorption = absorption_attention.squeeze()

        if predicted_absorption.device != device:
            predicted_absorption = predicted_absorption.to(device)

        # 处理batch size为1的情况
        if log_concentrations.numel() == 1 or predicted_absorption.numel() == 1:
            physics_loss_concentration = torch.tensor(0.0, device=device)
        else:
            try:
                # 使用更稳定的相关性计算
                correlation = torch.corrcoef(
                    torch.stack([log_concentrations, predicted_absorption])
                )[0, 1]
                if torch.isnan(correlation):
                    physics_loss_concentration = torch.tensor(0.0, device=device)
                else:
                    physics_loss_concentration = 1 - torch.abs(correlation)
            except:
                physics_loss_concentration = torch.tensor(0.0, device=device)

        # 3.4 污染等级与浓度的一致性约束
        predicted_class = torch.argmax(class_logits, dim=1).float()
        regression_values = regression.squeeze()

        # 处理batch size为1的情况
        if predicted_class.numel() == 1 or regression_values.numel() == 1:
            physics_loss_consistency = torch.tensor(0.0, device=device)
        else:
            try:
                correlation = torch.corrcoef(
                    torch.stack([predicted_class, regression_values])
                )[0, 1]
                if torch.isnan(correlation):
                    physics_loss_consistency = torch.tensor(0.0, device=device)
                else:
                    physics_loss_consistency = 1 - torch.abs(correlation)
            except:
                physics_loss_consistency = torch.tensor(0.0, device=device)

        # 组合物理损失
        physics_loss = (
            0.3 * physics_loss_spectral +
            0.3 * physics_loss_peaks +      # 峰值对齐
            0.2 * physics_loss_concentration +
            0.2 * physics_loss_consistency
        )

        # 总损失
        total_loss = (
            self.alpha * loss_cls +
            self.beta * loss_reg +
            self.gamma * physics_loss
        )

        return total_loss, {
            'loss_cls': loss_cls.item(),
            'loss_reg': loss_reg.item(),
            'loss_physics': physics_loss.item(),
            'loss_spectral': physics_loss_spectral.item(),
            'loss_peaks': physics_loss_peaks.item(),
            'loss_concentration': physics_loss_concentration.item(),
            'loss_consistency': physics_loss_consistency.item()
        }