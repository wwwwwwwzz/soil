"""
模型评估和可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path
import sys

sys.path.append('..')
from models.transformer import SpectralSpatialTransformer
from train import HyperspectralDataset
from torch.utils.data import DataLoader


def evaluate_model(checkpoint_path='checkpoints/best_model.pth'):
    """评估最佳模型"""

    # 加载模型
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']

    model = SpectralSpatialTransformer(
        num_bands=config['num_bands'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        num_classes=config['num_classes']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 加载测试数据
    data_path = Path('../data/processed')
    val_dataset = HyperspectralDataset(data_path, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 预测
    all_preds_cls = []
    all_labels_cls = []
    all_preds_reg = []
    all_labels_reg = []

    with torch.no_grad():
        for batch in val_loader:
            data, labels, concentrations = batch
            data = data.to(device)

            class_logits, regression = model(data)

            all_preds_cls.extend(torch.argmax(class_logits, dim=1).cpu().numpy())
            all_labels_cls.extend(labels.numpy())
            all_preds_reg.extend(regression.squeeze().cpu().numpy())
            all_labels_reg.extend(concentrations.numpy())

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 混淆矩阵
    cm = confusion_matrix(all_labels_cls, all_preds_cls)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_xlabel('预测类别')
    axes[0, 0].set_ylabel('真实类别')
    axes[0, 0].set_title('污染等级分类混淆矩阵')

    # 2. 浓度预测散点图
    axes[0, 1].scatter(all_labels_reg, all_preds_reg, alpha=0.5)
    axes[0, 1].plot([min(all_labels_reg), max(all_labels_reg)],
                    [min(all_labels_reg), max(all_labels_reg)],
                    'r--', lw=2)
    axes[0, 1].set_xlabel('真实浓度 (mg/kg)')
    axes[0, 1].set_ylabel('预测浓度 (mg/kg)')
    axes[0, 1].set_title('浓度预测结果')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 预测误差分布
    errors = np.array(all_preds_reg) - np.array(all_labels_reg)
    axes[1, 0].hist(errors, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('预测误差 (mg/kg)')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('预测误差分布')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 分类报告
    report = classification_report(all_labels_cls, all_preds_cls,
                                   target_names=[f'等级{i}' for i in range(5)],
                                   output_dict=True)

    # 显示分类报告
    report_text = classification_report(all_labels_cls, all_preds_cls,
                                        target_names=[f'等级{i}' for i in range(5)])
    axes[1, 1].text(0.1, 0.5, report_text, fontsize=10,
                    transform=axes[1, 1].transAxes,
                    verticalalignment='center',
                    fontfamily='monospace')
    axes[1, 1].set_title('分类报告')
    axes[1, 1].axis('off')

    plt.tight_layout()

    # 保存图像
    results_path = Path('results/figures')
    results_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_path / 'evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n评估结果已保存到: results/figures/evaluation_results.png")

    return all_preds_reg, all_labels_reg, all_preds_cls, all_labels_cls


if __name__ == "__main__":
    evaluate_model()