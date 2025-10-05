"""
模型训练脚本 - 修复版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
import gc
import warnings

warnings.filterwarnings('ignore')

import sys

sys.path.append('..')
from models.transformer import SpectralSpatialTransformer


class HyperspectralDataset(Dataset):
    """高光谱数据集 - 内存优化版"""

    def __init__(self, data_path, mode='train', use_memmap=True):
        self.mode = mode
        self.data_path = data_path

        # 使用内存映射来避免一次性加载所有数据
        if mode == 'train':
            if use_memmap:
                # 使用内存映射
                self.data = np.load(data_path / 'train_data.npy', mmap_mode='r')
            else:
                self.data = np.load(data_path / 'train_data.npy').astype(np.float32)
            self.labels = np.load(data_path / 'train_labels.npy')
            self.concentrations = np.load(data_path / 'train_concentrations.npy').astype(np.float32)
        else:
            if use_memmap:
                self.data = np.load(data_path / 'val_data.npy', mmap_mode='r')
            else:
                self.data = np.load(data_path / 'val_data.npy').astype(np.float32)
            self.labels = np.load(data_path / 'val_labels.npy')
            self.concentrations = np.load(data_path / 'val_concentrations.npy').astype(np.float32)

        print(f"{mode} 数据集: {len(self.data)} 个样本")

    def __len__(self):
        return len(self.labels)  # 使用labels的长度，避免访问大数组

    def __getitem__(self, idx):
        # 按需加载数据
        data_item = np.array(self.data[idx], dtype=np.float32)  # 转换为float32

        return (
            torch.from_numpy(data_item),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.concentrations[idx], dtype=torch.float32)
        )


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

        # 创建模型
        self.model = SpectralSpatialTransformer(
            num_bands=config['num_bands'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        ).to(self.device)

        # 打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数量: {total_params / 1e6:.2f}M")

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度器 - 修复：移除verbose参数
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
            # verbose参数已被移除
        )

        # 损失函数
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()

        # 记录训练历史
        self.train_history = {'loss': [], 'acc': [], 'r2': []}
        self.val_history = {'loss': [], 'acc': [], 'r2': []}

        # 最佳模型
        self.best_val_loss = float('inf')
        self.best_val_r2 = -float('inf')

        # 早停
        self.patience = config.get('patience', 15)
        self.patience_counter = 0

    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()

        total_loss = 0
        all_preds_cls = []
        all_labels_cls = []
        all_preds_reg = []
        all_labels_reg = []

        pbar = tqdm(dataloader, desc='训练')
        for batch_idx, batch in enumerate(pbar):
            try:
                data, labels, concentrations = batch
                data = data.to(self.device)
                labels = labels.to(self.device)
                concentrations = concentrations.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                class_logits, regression = self.model(data)

                # 计算损失
                loss_cls = self.criterion_cls(class_logits, labels)
                loss_reg = self.criterion_reg(regression.squeeze(), concentrations)

                # 规范化浓度损失（因为浓度值很大）
                loss_reg = loss_reg / 10000.0

                # 多任务损失
                loss = self.config['cls_weight'] * loss_cls + self.config['reg_weight'] * loss_reg

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # 记录
                total_loss += loss.item()
                all_preds_cls.extend(torch.argmax(class_logits, dim=1).cpu().numpy())
                all_labels_cls.extend(labels.cpu().numpy())
                all_preds_reg.extend(regression.squeeze().detach().cpu().numpy())
                all_labels_reg.extend(concentrations.cpu().numpy())

                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                # 定期清理显存
                if batch_idx % 50 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # 计算指标
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels_cls, all_preds_cls)

        # 处理R²计算
        try:
            r2 = r2_score(all_labels_reg, all_preds_reg)
        except:
            r2 = 0.0

        return avg_loss, accuracy, r2

    def validate(self, dataloader):
        """验证"""
        self.model.eval()

        total_loss = 0
        all_preds_cls = []
        all_labels_cls = []
        all_preds_reg = []
        all_labels_reg = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc='验证')
            for batch in pbar:
                data, labels, concentrations = batch
                data = data.to(self.device)
                labels = labels.to(self.device)
                concentrations = concentrations.to(self.device)

                # 前向传播
                class_logits, regression = self.model(data)

                # 计算损失
                loss_cls = self.criterion_cls(class_logits, labels)
                loss_reg = self.criterion_reg(regression.squeeze(), concentrations) / 10000.0
                loss = self.config['cls_weight'] * loss_cls + self.config['reg_weight'] * loss_reg

                # 记录
                total_loss += loss.item()
                all_preds_cls.extend(torch.argmax(class_logits, dim=1).cpu().numpy())
                all_labels_cls.extend(labels.cpu().numpy())
                all_preds_reg.extend(regression.squeeze().cpu().numpy())
                all_labels_reg.extend(concentrations.cpu().numpy())

                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 计算指标
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels_cls, all_preds_cls)

        try:
            r2 = r2_score(all_labels_reg, all_preds_reg)
            rmse = np.sqrt(mean_squared_error(all_labels_reg, all_preds_reg))
            mae = mean_absolute_error(all_labels_reg, all_preds_reg)
        except:
            r2 = 0.0
            rmse = 0.0
            mae = 0.0

        return avg_loss, accuracy, r2, rmse, mae, all_preds_reg, all_labels_reg

    def train(self, train_loader, val_loader, epochs):
        """完整训练流程"""
        print("\n开始训练...")
        print("=" * 60)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"学习率: {current_lr:.6f}")

            # 训练
            train_loss, train_acc, train_r2 = self.train_epoch(train_loader)
            self.train_history['loss'].append(train_loss)
            self.train_history['acc'].append(train_acc)
            self.train_history['r2'].append(train_r2)

            # 验证
            val_loss, val_acc, val_r2, val_rmse, val_mae, _, _ = self.validate(val_loader)
            self.val_history['loss'].append(val_loss)
            self.val_history['acc'].append(val_acc)
            self.val_history['r2'].append(val_r2)

            # 更新学习率
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"学习率调整: {old_lr:.6f} -> {new_lr:.6f}")

            # 打印结果
            print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}, R²: {train_r2:.4f}")
            print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, R²: {val_r2:.4f}")
            print(f"      RMSE: {val_rmse:.2f} mg/kg, MAE: {val_mae:.2f} mg/kg")

            # 保存最佳模型（基于R²分数）
            if val_r2 > self.best_val_r2:
                self.best_val_r2 = val_r2
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', epoch, val_loss, val_acc, val_r2)
                print("✅ 保存最佳模型 (R²)")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # 早停
            if self.patience_counter >= self.patience:
                print(f"\n早停: {self.patience} epochs没有改善")
                break

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_loss, val_acc, val_r2)

            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\n训练完成！")

        # 保存训练历史
        self.save_training_history()

        # 绘制图表
        try:
            self.plot_training_history()
        except Exception as e:
            print(f"绘图时出错: {e}")

    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'config': self.config
        }

        results_path = Path('../results')
        results_path.mkdir(exist_ok=True)

        import pickle
        with open(results_path / 'training_history.pkl', 'wb') as f:
            pickle.dump(history, f)

        print("训练历史已保存")

    def save_checkpoint(self, filename, epoch, val_loss, val_acc, val_r2):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_r2': val_r2,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        checkpoint_path = Path('../checkpoints')
        checkpoint_path.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path / filename)

    def plot_training_history(self):
        """绘制训练历史"""
        # 设置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 损失
        axes[0].plot(self.train_history['loss'], label='Train')
        axes[0].plot(self.val_history['loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 准确率
        axes[1].plot(self.train_history['acc'], label='Train')
        axes[1].plot(self.val_history['acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Classification Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # R²分数
        axes[2].plot(self.train_history['r2'], label='Train')
        axes[2].plot(self.val_history['r2'], label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('R²')
        axes[2].set_title('Concentration Prediction R²')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        results_path = Path('../results/figures')
        results_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(results_path / 'training_history.png', dpi=150)
        plt.close()
        print("训练历史图已保存")


def main():
    # 配置参数 - 优化内存使用
    config = {
        'num_bands': 512,
        'd_model': 128,  # 减小模型大小
        'n_heads': 4,  # 减少注意力头数
        'n_layers': 3,  # 减少层数
        'num_classes': 5,
        'dropout': 0.2,
        'batch_size': 16,  # 批量大小
        'learning_rate': 5e-4,
        'weight_decay': 1e-4,
        'epochs': 50,
        'patience': 10,
        'cls_weight': 0.3,  # 分类权重
        'reg_weight': 0.7  # 回归权重
    }

    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 数据路径
    data_path = Path('../data/processed')

    if not data_path.exists():
        print(f"错误: 数据路径不存在 {data_path}")
        print("请先运行数据预处理脚本")
        return

    # 创建数据集（使用内存映射）
    print("\n加载数据集...")
    train_dataset = HyperspectralDataset(data_path, mode='train', use_memmap=True)
    val_dataset = HyperspectralDataset(data_path, mode='val', use_memmap=True)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # 避免多进程内存问题
        pin_memory=False,
        drop_last=True  # 丢弃最后不完整的批次
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    # 创建训练器并训练
    trainer = Trainer(config)

    # 开始训练
    try:
        trainer.train(train_loader, val_loader, config['epochs'])
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()

    # 最终评估
    print("\n最终评估:")
    try:
        val_loss, val_acc, val_r2, val_rmse, val_mae, preds, labels = trainer.validate(val_loader)

        print(f"验证集结果:")
        print(f"  分类准确率: {val_acc:.4f}")
        print(f"  浓度预测R²: {val_r2:.4f}")
        print(f"  RMSE: {val_rmse:.2f} mg/kg")
        print(f"  MAE: {val_mae:.2f} mg/kg")

        # 保存预测结果
        results = {
            'predictions': preds,
            'ground_truth': labels,
            'metrics': {
                'accuracy': val_acc,
                'r2': val_r2,
                'rmse': val_rmse,
                'mae': val_mae
            }
        }

        results_path = Path('../results')
        results_path.mkdir(exist_ok=True)
        np.savez(results_path / 'predictions.npz', **results)
        print(f"\n预测结果已保存到: {results_path / 'predictions.npz'}")

    except Exception as e:
        print(f"评估时出错: {e}")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 如果使用GPU，设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main()