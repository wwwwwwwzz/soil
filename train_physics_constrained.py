"""
train_physics_constrained.py
训练物理约束模型 - 平衡权重版本
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, accuracy_score, classification_report
import sys
import time
from tqdm import tqdm

sys.path.append('..')

from models.physics_constrained_model import PhysicsConstrainedTransformer
from physics_loss import PhysicsConstrainedLoss
from interpretability_analysis import generate_interpretability_report
from experiments.train_improved import ImprovedHyperspectralDataset


def train_physics_model():
    # 配置 - 平衡分类和回归权重
    config = {
        'num_bands': 512,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'num_classes': 3,
        'dropout': 0.3,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'epochs': 150,
        'alpha': 0.45,  # 提高分类权重（原0.4）
        'beta': 0.45,   # 降低回归权重（原0.5）
        'gamma': 0.10   # 保持物理约束权重
    }

    print("🚀 物理约束模型训练开始（平衡权重版）")
    print("=" * 60)
    print("📊 训练配置:")
    print(f"  损失权重调整:")
    print(f"    分类权重 α: 0.40 → {config['alpha']} (↑)")
    print(f"    回归权重 β: 0.50 → {config['beta']} (↓)")
    print(f"    物理约束 γ: {config['gamma']} (不变)")
    print("-" * 60)
    for key, value in config.items():
        if key not in ['alpha', 'beta', 'gamma']:
            print(f"  {key:15s}: {value}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    # 数据加载
    data_path = Path('../data/processed')
    train_dataset = ImprovedHyperspectralDataset(data_path, mode='train')
    val_dataset = ImprovedHyperspectralDataset(data_path, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=0)

    print(f"📂 数据加载完成:")
    print(f"  训练样本: {len(train_dataset):,}")
    print(f"  验证样本: {len(val_dataset):,}")

    # 模型
    model = PhysicsConstrainedTransformer(
        num_bands=config['num_bands'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    ).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"🧠 模型信息:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

    # 损失和优化器 - 使用新的权重
    criterion = PhysicsConstrainedLoss(
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma']
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # 训练历史记录
    best_val_r2 = -float('inf')
    best_val_acc = 0
    best_epoch = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_r2': [], 'val_r2': [],
        'loss_breakdown': []
    }

    print(f"\n🏃‍♂️ 开始训练 ({config['epochs']} epochs)")
    print("=" * 80)

    # 训练循环
    for epoch in range(config['epochs']):
        start_time = time.time()

        # ========== 训练阶段 ==========
        model.train()
        train_losses = []
        train_accs = []
        train_preds = []
        train_labels = []
        train_regs = []
        train_reg_targets = []
        epoch_loss_breakdown = {
            'cls': [], 'reg': [], 'physics': [],
            'spectral': [], 'concentration': [], 'consistency': []
        }

        # 训练进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{config['epochs']} [Train]",
                         leave=False, ncols=100)

        for batch_idx, batch in enumerate(train_pbar):
            data, labels, norm_conc, orig_conc, _ = batch
            data = data.to(device)
            labels = labels.to(device)
            norm_conc = norm_conc.to(device)
            orig_conc = orig_conc.to(device)

            optimizer.zero_grad()
            outputs = model(data, return_attention=True)
            targets = (labels, norm_conc, orig_conc)

            loss, loss_dict = criterion(outputs, targets, model)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录指标
            train_losses.append(loss.item())
            class_preds = torch.argmax(outputs[0], dim=1)
            train_accs.append((class_preds == labels).float().mean().item())

            train_preds.extend(outputs[1].squeeze().detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_regs.extend(outputs[1].squeeze().detach().cpu().numpy())
            train_reg_targets.extend(norm_conc.cpu().numpy())

            # 记录损失分解
            for key in epoch_loss_breakdown:
                if f'loss_{key}' in loss_dict:
                    epoch_loss_breakdown[key].append(loss_dict[f'loss_{key}'])

            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_accs[-1]:.3f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # ========== 验证阶段 ==========
        model.eval()
        val_losses = []
        val_accs = []
        val_preds_class = []
        val_preds_reg = []
        val_labels = []
        val_reg_targets = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:3d}/{config['epochs']} [Val]  ",
                       leave=False, ncols=100)

        with torch.no_grad():
            for batch in val_pbar:
                data, labels, norm_conc, orig_conc, _ = batch
                data = data.to(device)
                labels_dev = labels.to(device)
                norm_conc_dev = norm_conc.to(device)
                orig_conc_dev = orig_conc.to(device)

                outputs = model(data, return_attention=True)
                targets = (labels_dev, norm_conc_dev, orig_conc_dev)

                loss, _ = criterion(outputs, targets, model)

                val_losses.append(loss.item())
                class_preds = torch.argmax(outputs[0], dim=1)
                val_accs.append((class_preds == labels_dev).float().mean().item())

                val_preds_class.extend(class_preds.cpu().numpy())
                val_preds_reg.extend(outputs[1].squeeze().cpu().numpy())
                val_labels.extend(labels.numpy())
                val_reg_targets.extend(norm_conc.numpy())

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{val_accs[-1]:.3f}'
                })

        # 计算epoch指标
        train_loss_avg = np.mean(train_losses)
        val_loss_avg = np.mean(val_losses)
        train_acc_avg = np.mean(train_accs)
        val_acc_avg = accuracy_score(val_labels, val_preds_class)

        # 计算R²
        train_r2 = r2_score(train_reg_targets, train_regs) if len(train_regs) > 1 else 0
        val_r2 = r2_score(val_reg_targets, val_preds_reg) if len(val_preds_reg) > 1 else 0

        # 记录历史
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['train_acc'].append(train_acc_avg)
        history['val_acc'].append(val_acc_avg)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)

        # 计算平均损失分解
        avg_loss_breakdown = {k: np.mean(v) if v else 0 for k, v in epoch_loss_breakdown.items()}
        history['loss_breakdown'].append(avg_loss_breakdown)

        # 学习率调整
        scheduler.step()

        # 计算训练时间
        epoch_time = time.time() - start_time

        # 打印详细信息
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n📊 Epoch {epoch+1:3d}/{config['epochs']} - {epoch_time:.1f}s")
            print(f"  🏋️ 训练 | Loss: {train_loss_avg:.4f} | Acc: {train_acc_avg:.3f} | R²: {train_r2:.4f}")
            print(f"  🎯 验证 | Loss: {val_loss_avg:.4f} | Acc: {val_acc_avg:.3f} | R²: {val_r2:.4f}")

            # 显示物理损失分解
            if avg_loss_breakdown:
                print(f"  🔬 损失分解:")
                print(f"     分类: {avg_loss_breakdown.get('cls', 0):.4f} ({config['alpha']*100:.0f}%权重)")
                print(f"     回归: {avg_loss_breakdown.get('reg', 0):.4f} ({config['beta']*100:.0f}%权重)")
                print(f"     物理: {avg_loss_breakdown.get('physics', 0):.4f} ({config['gamma']*100:.0f}%权重)")

            print(f"  📈 学习率: {optimizer.param_groups[0]['lr']:.2e}")

            # 保存最佳模型 - 综合考虑准确率和R²
            # 使用加权评分：0.5 * acc + 0.5 * r2
            val_score = 0.5 * val_acc_avg + 0.5 * val_r2
            best_score = 0.5 * best_val_acc + 0.5 * best_val_r2

            if val_score > best_score:
                best_val_r2 = val_r2
                best_val_acc = val_acc_avg
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_r2': val_r2,
                    'val_acc': val_acc_avg,
                    'config': config,
                    'history': history
                }, 'best_physics_model_balanced.pth')
                print(f"  ✅ 新的最佳模型已保存 (综合评分: {val_score:.4f})")

            print("-" * 80)

    # 训练完成
    print(f"\n🎉 训练完成!")
    print(f"📈 最佳验证结果 (Epoch {best_epoch+1}):")
    print(f"   准确率: {best_val_acc:.4f}")
    print(f"   R²: {best_val_r2:.4f}")
    print(f"   综合评分: {0.5 * best_val_acc + 0.5 * best_val_r2:.4f}")

    # 加载最佳模型进行最终评估
    checkpoint = torch.load('best_physics_model_balanced.pth', weights_only=False)  # 添加weights_only=False
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 最终验证集评估
    print(f"\n🔍 最终模型评估:")
    final_preds_class = []
    final_preds_reg = []
    final_labels = []
    final_reg_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final Evaluation"):
            data, labels, norm_conc, orig_conc, _ = batch
            data = data.to(device)
            labels_dev = labels.to(device)
            norm_conc_dev = norm_conc.to(device)

            outputs = model(data, return_attention=True)

            class_preds = torch.argmax(outputs[0], dim=1)
            final_preds_class.extend(class_preds.cpu().numpy())
            final_labels.extend(labels.numpy())
            final_preds_reg.extend(outputs[1].squeeze().cpu().numpy())
            final_reg_targets.extend(norm_conc.numpy())

    final_acc = accuracy_score(final_labels, final_preds_class)
    final_r2 = r2_score(final_reg_targets, final_preds_reg)

    print(f"  分类准确率: {final_acc:.4f}")
    print(f"  回归R²:    {final_r2:.4f}")
    print(f"  综合评分:   {0.5 * final_acc + 0.5 * final_r2:.4f}")

    # 分类报告
    print(f"\n📋 详细分类报告:")
    class_names = ['清洁 (0)', '轻度污染 (3)', '重度污染 (4)']
    print(classification_report(final_labels, final_preds_class,
                              target_names=class_names, digits=4))

    # 与基线对比
    print("\n📊 与基线模型对比:")
    baseline_results = {
        'Standard Transformer': {'acc': 0.9688, 'r2': 0.9337},
        'SVM': {'acc': 0.9628, 'r2': 0.8652},
        'PLSR': {'acc': 0.9400, 'r2': 0.8517}
    }

    for model_name, results in baseline_results.items():
        print(f"  {model_name:20s}: Acc={results['acc']:.4f}, R²={results['r2']:.4f}")
    print(f"  {'Physics-Constrained':20s}: Acc={final_acc:.4f}, R²={final_r2:.4f} ← 新权重")

    # 计算提升
    print("\n🎯 性能提升分析:")
    st_acc_improve = (final_acc - 0.9688) / 0.9688 * 100
    st_r2_improve = (final_r2 - 0.9337) / 0.9337 * 100
    print(f"  相比Standard Transformer:")
    print(f"    准确率变化: {st_acc_improve:+.2f}%")
    print(f"    R²变化:    {st_r2_improve:+.2f}%")

    if final_acc >= 0.96 and final_r2 >= 0.94:
        print("\n✨ 优秀！模型在保持R²优势的同时达到了高准确率！")
        print("📝 这个结果完全符合SCI论文发表要求")

    # 生成物理可解释性报告
    print(f"\n🔬 生成物理可解释性报告...")
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True)

    try:
        scores = generate_interpretability_report(model, val_loader, save_dir='../results/')
        print(f"📊 物理一致性总分: {scores.get('Overall', 0):.3f}")
        print(f"📁 报告已保存到: {results_dir.absolute()}")
    except Exception as e:
        print(f"⚠️ 可解释性分析出错: {e}")
        scores = None

    return model, scores


if __name__ == "__main__":
    model, scores = train_physics_model()