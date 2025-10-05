"""
改进的训练脚本 - 标签映射修复版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from models.transformer import SpectralSpatialTransformer

class ImprovedHyperspectralDataset(Dataset):
    """改进的数据集 - 包含标签映射和浓度归一化"""
    def __init__(self, data_path, mode='train', use_memmap=True):
        self.mode = mode

        # 加载数据
        if mode == 'train':
            self.data = np.load(data_path / 'train_data.npy', mmap_mode='r' if use_memmap else None)
            self.original_labels = np.load(data_path / 'train_labels.npy')
            self.concentrations = np.load(data_path / 'train_concentrations.npy').astype(np.float32)
        else:
            self.data = np.load(data_path / 'val_data.npy', mmap_mode='r' if use_memmap else None)
            self.original_labels = np.load(data_path / 'val_labels.npy')
            self.concentrations = np.load(data_path / 'val_concentrations.npy').astype(np.float32)

        # 创建标签映射
        unique_labels = np.unique(self.original_labels)
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        self.inverse_mapping = {v: k for k, v in self.label_mapping.items()}

        # 应用标签映射
        self.labels = np.array([self.label_mapping[label] for label in self.original_labels])

        print(f"{mode} 数据集: {len(self.data)} 个样本")
        print(f"  原始标签: {unique_labels}")
        print(f"  标签映射: {self.label_mapping}")
        print(f"  新标签范围: {self.labels.min()} - {self.labels.max()}")

        # 对浓度进行对数变换和标准化
        self.log_concentrations = np.log1p(self.concentrations)

        # 计算或加载标准化参数
        if mode == 'train':
            self.mean = self.log_concentrations.mean()
            self.std = self.log_concentrations.std()
            # 保存标准化参数和标签映射
            params = {
                'mean': self.mean,
                'std': self.std,
                'label_mapping': self.label_mapping,
                'inverse_mapping': self.inverse_mapping
            }
            with open(data_path / 'preprocessing_params.pkl', 'wb') as f:
                pickle.dump(params, f)
        else:
            # 加载训练集的参数
            try:
                with open(data_path / 'preprocessing_params.pkl', 'rb') as f:
                    params = pickle.load(f)
                    self.mean = params['mean']
                    self.std = params['std']
            except:
                self.mean = self.log_concentrations.mean()
                self.std = self.log_concentrations.std()

        # 标准化
        self.normalized_concentrations = (self.log_concentrations - self.mean) / (self.std + 1e-8)

        print(f"  浓度范围: {self.concentrations.min():.2f} - {self.concentrations.max():.2f} mg/kg")
        print(f"  标准化后: {self.normalized_concentrations.min():.2f} - {self.normalized_concentrations.max():.2f}")

        # 计算类别权重
        unique_mapped_labels, counts = np.unique(self.labels, return_counts=True)
        max_count = counts.max()
        self.class_weights_dict = {}

        print(f"  类别分布:")
        for label, count in zip(unique_mapped_labels, counts):
            weight = max_count / count
            self.class_weights_dict[label] = weight
            orig_label = self.inverse_mapping[label]
            percentage = count / len(self.labels) * 100
            print(f"    等级 {orig_label} (映射为{label}): {count} 个样本 ({percentage:.1f}%), 权重: {weight:.2f}")

        # 为每个样本分配权重
        self.sample_weights = np.array([self.class_weights_dict[label] for label in self.labels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_item = np.array(self.data[idx], dtype=np.float32)

        return (
            torch.from_numpy(data_item),
            torch.tensor(self.labels[idx], dtype=torch.long),  # 使用映射后的标签
            torch.tensor(self.normalized_concentrations[idx], dtype=torch.float32),
            torch.tensor(self.concentrations[idx], dtype=torch.float32),
            torch.tensor(self.original_labels[idx], dtype=torch.long)  # 保留原始标签用于显示
        )

    def denormalize(self, normalized_values):
        """反归一化预测值"""
        log_values = normalized_values * self.std + self.mean
        return np.expm1(log_values)

class ImprovedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # 创建模型
        self.model = SpectralSpatialTransformer(
            num_bands=config['num_bands'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数量: {total_params/1e6:.2f}M")

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        # 损失函数
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()  # 改回MSE

        # 记录
        self.train_history = {'loss': [], 'acc': [], 'r2': [], 'rmse': []}
        self.val_history = {'loss': [], 'acc': [], 'r2': [], 'rmse': []}

        self.best_val_r2 = -float('inf')
        self.best_val_acc = 0
        self.patience_counter = 0

    def train_epoch(self, dataloader, dataset):
        self.model.train()

        losses = []
        all_preds_cls = []
        all_labels_cls = []
        all_preds_reg_normalized = []
        all_labels_reg_original = []

        pbar = tqdm(dataloader, desc='训练')
        for batch in pbar:
            try:
                data, labels, norm_concentrations, orig_concentrations, _ = batch
                data = data.to(self.device)
                labels = labels.to(self.device)
                norm_concentrations = norm_concentrations.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                class_logits, regression = self.model(data)

                # 损失计算
                loss_cls = self.criterion_cls(class_logits, labels)
                loss_reg = self.criterion_reg(regression.squeeze(), norm_concentrations)

                # 组合损失
                loss = self.config['cls_weight'] * loss_cls + self.config['reg_weight'] * loss_reg

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 记录
                losses.append(loss.item())
                all_preds_cls.extend(torch.argmax(class_logits, dim=1).cpu().numpy())
                all_labels_cls.extend(labels.cpu().numpy())
                all_preds_reg_normalized.extend(regression.squeeze().detach().cpu().numpy())
                all_labels_reg_original.extend(orig_concentrations.cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            except Exception as e:
                print(f'批次处理错误: {e}')
                continue

        # 反归一化预测值
        all_preds_reg = dataset.denormalize(np.array(all_preds_reg_normalized))

        # 计算指标
        avg_loss = np.mean(losses) if losses else 0
        accuracy = accuracy_score(all_labels_cls, all_preds_cls) if all_labels_cls else 0

        try:
            r2 = r2_score(all_labels_reg_original, all_preds_reg)
        except:
            r2 = -1.0

        rmse = np.sqrt(mean_squared_error(all_labels_reg_original, all_preds_reg)) if all_labels_reg_original else 0

        return avg_loss, accuracy, r2, rmse

    def validate(self, dataloader, dataset):
        self.model.eval()

        losses = []
        all_preds_cls = []
        all_labels_cls = []
        all_preds_reg_normalized = []
        all_labels_reg_original = []
        all_original_labels = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc='验证')
            for batch in pbar:
                data, labels, norm_concentrations, orig_concentrations, orig_labels = batch
                data = data.to(self.device)
                labels = labels.to(self.device)
                norm_concentrations = norm_concentrations.to(self.device)

                # 前向传播
                class_logits, regression = self.model(data)

                # 损失
                loss_cls = self.criterion_cls(class_logits, labels)
                loss_reg = self.criterion_reg(regression.squeeze(), norm_concentrations)
                loss = self.config['cls_weight'] * loss_cls + self.config['reg_weight'] * loss_reg

                # 记录
                losses.append(loss.item())
                all_preds_cls.extend(torch.argmax(class_logits, dim=1).cpu().numpy())
                all_labels_cls.extend(labels.cpu().numpy())
                all_preds_reg_normalized.extend(regression.squeeze().cpu().numpy())
                all_labels_reg_original.extend(orig_concentrations.cpu().numpy())
                all_original_labels.extend(orig_labels.cpu().numpy())

        # 反归一化
        all_preds_reg = dataset.denormalize(np.array(all_preds_reg_normalized))

        # 指标
        avg_loss = np.mean(losses) if losses else 0
        accuracy = accuracy_score(all_labels_cls, all_preds_cls) if all_labels_cls else 0

        try:
            r2 = r2_score(all_labels_reg_original, all_preds_reg)
        except:
            r2 = -1.0

        rmse = np.sqrt(mean_squared_error(all_labels_reg_original, all_preds_reg)) if all_labels_reg_original else 0
        mae = mean_absolute_error(all_labels_reg_original, all_preds_reg) if all_labels_reg_original else 0

        return avg_loss, accuracy, r2, rmse, mae, all_preds_reg, all_labels_reg_original

    def train(self, train_loader, val_loader, train_dataset, val_dataset, epochs):
        print("\n开始训练...")
        print("="*60)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # 训练
            train_loss, train_acc, train_r2, train_rmse = self.train_epoch(train_loader, train_dataset)
            self.train_history['loss'].append(train_loss)
            self.train_history['acc'].append(train_acc)
            self.train_history['r2'].append(train_r2)
            self.train_history['rmse'].append(train_rmse)

            # 验证
            val_loss, val_acc, val_r2, val_rmse, val_mae, _, _ = self.validate(val_loader, val_dataset)
            self.val_history['loss'].append(val_loss)
            self.val_history['acc'].append(val_acc)
            self.val_history['r2'].append(val_r2)
            self.val_history['rmse'].append(val_rmse)

            # 学习率调度
            self.scheduler.step()

            # 打印结果
            print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}, R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}")
            print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, R²: {val_r2:.4f}")
            print(f"      RMSE: {val_rmse:.2f} mg/kg, MAE: {val_mae:.2f} mg/kg")

            # 保存最佳模型
            if val_r2 > self.best_val_r2:
                self.best_val_r2 = val_r2
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model_improved.pth', epoch, val_loss, val_acc, val_r2)
                print(f"✅ 保存最佳模型 (R²={val_r2:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # 早停
            if self.patience_counter >= self.config['patience']:
                print(f"\n早停: {self.config['patience']} epochs没有改善")
                break

        print("\n训练完成！")

    def save_checkpoint(self, filename, epoch, val_loss, val_acc, val_r2):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_r2': val_r2,
            'config': self.config
        }

        checkpoint_path = Path('../checkpoints')
        checkpoint_path.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path / filename)

def main():
    config = {
        'num_bands': 512,
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'num_classes': 3,  # 实际只有3个不同的类
        'dropout': 0.3,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'weight_decay': 1e-3,
        'epochs': 50,
        'patience': 15,
        'cls_weight': 0.4,
        'reg_weight': 0.6
    }

    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 创建数据集
    data_path = Path('../data/processed')
    print("\n加载数据集...")
    train_dataset = ImprovedHyperspectralDataset(data_path, mode='train')
    val_dataset = ImprovedHyperspectralDataset(data_path, mode='val')

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"\n数据加载器:")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")

    # 训练
    trainer = ImprovedTrainer(config)

    try:
        trainer.train(train_loader, val_loader, train_dataset, val_dataset, config['epochs'])
    except KeyboardInterrupt:
        print("\n训练被中断")

    # 最终评估
    print("\n最终评估:")
    val_loss, val_acc, val_r2, val_rmse, val_mae, preds, labels = trainer.validate(val_loader, val_dataset)

    print(f"\n最终结果:")
    print(f"  分类准确率: {val_acc:.4f}")
    print(f"  浓度R²: {val_r2:.4f}")
    print(f"  RMSE: {val_rmse:.2f} mg/kg")
    print(f"  MAE: {val_mae:.2f} mg/kg")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    main()