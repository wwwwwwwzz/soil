"""
run_interpretability_analysis.py
运行物理可解释性分析的独立脚本
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import os

# 添加项目路径
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./experiments')


def main():
    print("=" * 60)
    print("高光谱土壤污染检测 - 物理可解释性分析")
    print("=" * 60)

    # 检查依赖
    try:
        from models.physics_constrained_model import PhysicsConstrainedTransformer
        from experiments.train_improved import ImprovedHyperspectralDataset
        from interpretability_analysis import generate_interpretability_report
        print("✓ 模块导入成功")
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请检查文件路径和依赖安装")
        return

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ 使用设备: {device}")

    # 检查模型文件
    model_paths = [
        'best_physics_model.pth',
        'checkpoints/best_physics_model.pth',
        'experiments/best_physics_model.pth'
    ]

    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break

    if model_path is None:
        print("✗ 找不到训练好的模型文件")
        print("请先运行以下命令训练模型：")
        print("  python experiments/train_physics_constrained.py")
        return

    print(f"✓ 找到模型文件: {model_path}")

    # 加载模型
    try:
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {
            'num_bands': 512,
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'num_classes': 3,
            'dropout': 0.3
        })

        model = PhysicsConstrainedTransformer(
            num_bands=config['num_bands'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        val_r2 = checkpoint.get('val_r2', 'Unknown')
        print(f"✓ 模型加载成功 (验证集R²: {val_r2})")

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return

    # 检查数据文件
    data_paths = [
        'data/processed',
        '../data/processed'
    ]

    data_path = None
    for path in data_paths:
        if Path(path).exists():
            data_path = Path(path)
            break

    if data_path is None:
        print("✗ 找不到预处理数据目录")
        print("请先运行数据预处理脚本")
        return

    print(f"✓ 找到数据目录: {data_path}")

    # 加载验证数据
    try:
        val_dataset = ImprovedHyperspectralDataset(data_path, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        print(f"✓ 验证数据加载成功 (样本数: {len(val_dataset)})")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return

    # 创建结果目录
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # 运行分析
    print("\n开始物理可解释性分析...")
    print("-" * 40)

    try:
        scores = generate_interpretability_report(model, val_loader, save_dir='results/')

        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)

        print("\n📊 生成的文件：")
        print("  📈 results/physics_analysis.png  - 可视化分析图表")
        print("  📄 results/physics_report.txt   - 详细分析报告")

        print(f"\n📋 物理一致性评分：")
        print("-" * 30)
        for metric, score in scores.items():
            status = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
            print(f"  {status} {metric:20s}: {score:.4f}")

        print(f"\n💡 总体评价：")
        overall_score = scores.get('Overall', 0)
        if overall_score > 0.8:
            print("  🎉 优秀！模型展现出很强的物理一致性")
        elif overall_score > 0.6:
            print("  👍 良好！模型具有合理的物理可解释性")
        else:
            print("  ⚠️  需要改进物理约束机制")

    except Exception as e:
        print(f"✗ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()