"""
quick_analysis.py
修正路径的快速可解释性分析脚本
放在experiments目录下运行
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent  # 从experiments回到项目根目录
sys.path.append(str(project_root))


def run_analysis():
    print("🚀 开始物理可解释性分析...")
    print(f"📁 项目根目录: {project_root.absolute()}")

    # 确保results目录存在（与experiments平级）
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    print(f"✓ 结果目录: {results_dir.absolute()}")

    try:
        # 导入模块
        from models.physics_constrained_model import PhysicsConstrainedTransformer
        from interpretability_analysis import generate_interpretability_report
        from experiments.train_improved import ImprovedHyperspectralDataset
        print("✓ 模块导入成功")
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return

    # 加载最佳模型（在experiments目录下）
    model_path = Path(__file__).parent / 'best_physics_model.pth'
    if not model_path.exists():
        print(f"❌ 找不到模型文件: {model_path}")
        print("请确保已经运行完训练脚本")
        return

    print(f"✓ 找到模型: {model_path}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ 使用设备: {device}")

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    val_r2 = checkpoint.get('val_r2', 'Unknown')

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

    print(f"✓ 模型加载成功")
    print(f"  - 验证R²: {val_r2:.4f}")
    print(f"  - 模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 加载数据（在项目根目录下）
    data_path = project_root / 'data' / 'processed'
    if not data_path.exists():
        print(f"❌ 找不到数据目录: {data_path}")
        return

    val_dataset = ImprovedHyperspectralDataset(data_path, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"✓ 数据加载成功 (验证样本: {len(val_dataset)})")

    # 运行分析
    print("\n📊 生成物理可解释性报告...")
    print("-" * 50)

    try:
        scores = generate_interpretability_report(
            model,
            val_loader,
            save_dir=str(results_dir) + '/'
        )

        print("\n" + "=" * 60)
        print("🎉 物理可解释性分析完成！")
        print("=" * 60)

        print(f"\n📈 生成的文件：")
        print(f"  📊 {results_dir}/physics_analysis.png  - 物理分析可视化")
        print(f"  📄 {results_dir}/physics_report.txt   - 详细分析报告")

        print(f"\n🔬 物理一致性评分：")
        print("-" * 40)
        for metric, score in scores.items():
            if score > 0.8:
                status = "🟢 优秀"
            elif score > 0.6:
                status = "🟡 良好"
            else:
                status = "🔴 需改进"
            print(f"  {status} {metric:20s}: {score:.4f}")

        overall_score = scores.get('Overall', 0)
        print(f"\n💡 总体评价：")
        if overall_score > 0.8:
            print("  🏆 卓越！模型展现出强大的物理一致性")
            print("  📝 结果完全满足SCI二区论文发表要求")
            print("  🎯 可以重点突出Physics-Informed AI的创新性")
        elif overall_score > 0.6:
            print("  👍 良好！模型具有合理的物理可解释性")
            print("  📈 可以考虑进一步优化物理约束参数")
        else:
            print("  ⚠️  物理约束效果有待改进")
            print("  🔧 建议调整gamma参数或优化物理损失函数")

        print(f"\n🎓 论文写作建议：")
        print("  - 强调物理先验知识的融入方法")
        print("  - 对比传统黑箱方法的优势")
        print("  - 展示Beer-Lambert定律约束的有效性")
        print("  - 突出多任务学习的物理一致性")

        return scores

    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    scores = run_analysis()
    if scores:
        print(
            f"\n✨ 分析成功完成！您的物理约束模型表现{['需要改进', '良好', '优秀'][min(2, int(scores.get('Overall', 0) * 3))]}")