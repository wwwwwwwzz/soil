"""
数据预处理模块 - 内存优化版（分批处理，避免内存溢出）
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tifffile as tiff
from scipy import signal
from scipy.spatial import ConvexHull
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc  # 垃圾回收

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DataProcessor:
    def __init__(self):
        self.data_path = Path("data/raw/hyperspectral")
        self.output_path = Path("data/processed")
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 固定的波段参数
        self.num_bands = 512
        self.wavelength_start = 889
        self.wavelength_end = 1710
        self.wavelengths = np.linspace(self.wavelength_start, self.wavelength_end, self.num_bands)

    def prepare_data(self):
        """主函数：准备所有数据"""
        print("=" * 50)
        print("加载21个培养皿的高光谱TIF数据...")
        print(f"数据格式: (512波段, 高度, 宽度)")
        print("=" * 50)

        # 1. 加载原始数据
        all_data = []
        all_labels = []
        all_concentrations = []

        # 读取标签文件
        try:
            labels_df = pd.read_excel("data/raw/labels.xlsx")
            print(f"✅ 成功加载标签文件，共{len(labels_df)}个样本")
        except Exception as e:
            print(f"⚠️ 加载标签文件失败: {e}")
            print("创建示例标签文件...")
            self.create_example_labels()
            labels_df = pd.read_excel("data/raw/labels.xlsx")

        # 记录每个文件的信息
        file_info = []

        # 逐个加载培养皿数据
        for i in range(1, 15):
            file_path = self.data_path / f"sample_{i:02d}.tif"

            if not file_path.exists():
                print(f"⚠️ 文件不存在: {file_path}")
                continue

            try:
                # 加载TIF文件
                print(f"\n处理培养皿 {i}...")
                hyperspectral = self.load_tif_file(file_path)

                print(f"  原始尺寸: (512波段, {hyperspectral.shape[0]}高, {hyperspectral.shape[1]}宽)")

                # 获取对应的浓度标签
                concentration = labels_df.loc[i - 1, 'concentration']
                pollution_level = labels_df.loc[
                    i - 1, 'pollution_level'] if 'pollution_level' in labels_df.columns else self.get_pollution_level(
                    concentration)

                # 10×10分割
                patches = self.split_into_patches(hyperspectral, patch_size=10)

                print(f"  生成 {len(patches)} 个10×10样本")
                print(f"  浓度: {concentration:.2f} mg/kg")
                print(f"  污染等级: {pollution_level}")

                # 记录文件信息
                file_info.append({
                    'sample_id': i,
                    'height': hyperspectral.shape[0],
                    'width': hyperspectral.shape[1],
                    'n_patches': len(patches),
                    'concentration': concentration,
                    'pollution_level': pollution_level
                })

                all_data.extend(patches)
                all_labels.extend([pollution_level] * len(patches))
                all_concentrations.extend([concentration] * len(patches))

            except Exception as e:
                print(f"❌ 处理培养皿 {i} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(all_data) == 0:
            print("❌ 没有成功加载任何数据！")
            return

        # 保存文件信息
        pd.DataFrame(file_info).to_csv(self.output_path / "file_info.csv", index=False)

        # 2. 转换为数组
        print(f"\n转换数据格式...")
        all_data = np.array(all_data, dtype=np.float32)
        all_labels = np.array(all_labels)
        all_concentrations = np.array(all_concentrations)

        print(f"数据形状: {all_data.shape}")
        print(f"  样本数: {all_data.shape[0]}")
        print(f"  空间尺寸: {all_data.shape[1]}×{all_data.shape[2]}")
        print(f"  波段数: {all_data.shape[3]}")
        print(f"  内存占用: {all_data.nbytes / 1024 / 1024 / 1024:.2f} GB")

        # 统计标签分布
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        print(f"\n标签分布:")
        for label, count in zip(unique_labels, counts):
            print(f"  等级 {label}: {count} 个样本 ({count / len(all_labels) * 100:.1f}%)")

        # 3. 内存优化的预处理
        print("\n应用预处理（内存优化版）...")
        processed_data = self.preprocess_pipeline_memory_efficient(all_data)

        # 4. 数据分割
        n_samples = len(processed_data)

        print("\n分层采样...")
        train_indices = []
        val_indices = []

        for label in unique_labels:
            label_indices = np.where(all_labels == label)[0]
            np.random.shuffle(label_indices)

            n_label_train = int(len(label_indices) * 0.8)
            if n_label_train == 0 and len(label_indices) > 0:
                n_label_train = 1

            train_indices.extend(label_indices[:n_label_train])
            val_indices.extend(label_indices[n_label_train:])

            print(f"  等级 {label}: {n_label_train} 训练, {len(label_indices) - n_label_train} 验证")

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        # 5. 保存处理后的数据
        print("\n保存处理后的数据...")

        # 分批保存以避免内存问题
        np.save(self.output_path / "train_data.npy", processed_data[train_indices].astype(np.float16))  # 使用float16节省空间
        np.save(self.output_path / "train_labels.npy", all_labels[train_indices])
        np.save(self.output_path / "train_concentrations.npy", all_concentrations[train_indices])

        np.save(self.output_path / "val_data.npy", processed_data[val_indices].astype(np.float16))
        np.save(self.output_path / "val_labels.npy", all_labels[val_indices])
        np.save(self.output_path / "val_concentrations.npy", all_concentrations[val_indices])

        # 保存完整数据（可选，如果内存允许）
        # np.save(self.output_path / "all_data.npy", processed_data.astype(np.float16))
        # np.save(self.output_path / "all_labels.npy", all_labels)
        # np.save(self.output_path / "all_concentrations.npy", all_concentrations)

        # 保存元数据
        metadata = {
            'num_bands': self.num_bands,
            'wavelength_range': [self.wavelength_start, self.wavelength_end],
            'n_train': len(train_indices),
            'n_val': len(val_indices),
            'n_total': n_samples,
            'label_distribution': {int(label): int(count) for label, count in zip(unique_labels, counts)}
        }

        import json
        with open(self.output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 50)
        print("✅ 数据准备完成！")
        print(f"  总样本数: {n_samples}")
        print(f"  训练集: {len(train_indices)} 个样本")
        print(f"  验证集: {len(val_indices)} 个样本")
        print(f"  波段数: {self.num_bands}")
        print(f"  数据保存在: {self.output_path}")
        print("=" * 50)

    def preprocess_pipeline_memory_efficient(self, data):
        """内存优化的预处理流程"""

        # 1. 数据标准化（原地操作）
        print("  1. 数据标准化...")
        self.normalize_data_inplace(data)

        # 2. 简化的连续统去除（可选）
        print("  2. 跳过连续统去除（节省内存）...")
        # 如果需要，可以只对部分样本做

        # 3. 简化的Savitzky-Golay滤波
        print("  3. Savitzky-Golay滤波（使用scipy的axis参数）...")
        data = self.savgol_filter_axis(data)

        # 4. SNV标准化（分批处理）
        print("  4. SNV标准化（分批处理）...")
        self.snv_normalization_batch(data)

        return data

    def normalize_data_inplace(self, data):
        """原地数据标准化（节省内存）"""
        for i in tqdm(range(len(data)), desc="    标准化"):
            sample = data[i]
            min_val = sample.min()
            max_val = sample.max()
            if max_val > min_val:
                data[i] = (sample - min_val) / (max_val - min_val)

    def savgol_filter_axis(self, data):
        """使用axis参数的Savitzky-Golay滤波（最快）"""
        print(f"    对 {len(data)} 个样本进行滤波...")

        # 直接对最后一个轴（光谱维度）进行滤波
        filtered_data = signal.savgol_filter(
            data,
            window_length=11,
            polyorder=3,
            axis=-1,  # 对最后一个轴（光谱维度）进行滤波
            mode='nearest'  # 边界处理
        )

        return filtered_data

    def snv_normalization_batch(self, data, batch_size=1000):
        """分批SNV标准化（避免内存溢出）"""
        n_samples = len(data)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="    SNV标准化批次"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)

            # 处理这一批
            batch = data[start_idx:end_idx]
            batch_shape = batch.shape

            # 重塑为 (batch_size*h*w, bands)
            batch_reshaped = batch.reshape(-1, self.num_bands)

            # 计算均值和标准差
            means = batch_reshaped.mean(axis=1, keepdims=True)
            stds = batch_reshaped.std(axis=1, keepdims=True)

            # 避免除零
            stds[stds == 0] = 1

            # SNV标准化（原地操作）
            batch_reshaped[:] = (batch_reshaped - means) / stds

            # 重塑回原始形状并写回
            data[start_idx:end_idx] = batch_reshaped.reshape(batch_shape)

            # 释放临时变量
            del batch, batch_reshaped, means, stds

            # 定期垃圾回收
            if batch_idx % 10 == 0:
                gc.collect()

    def load_tif_file(self, file_path):
        """加载TIF文件"""
        img = tiff.imread(str(file_path))

        # ENVI格式: (512 bands, height, width) -> (height, width, 512 bands)
        if img.shape[0] == 512:
            img = np.transpose(img, (1, 2, 0))

        return img

    def split_into_patches(self, image, patch_size=10):
        """分割成小块"""
        patches = []
        h, w, bands = image.shape

        n_patches_h = h // patch_size
        n_patches_w = w // patch_size

        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = image[
                        i * patch_size:(i + 1) * patch_size,
                        j * patch_size:(j + 1) * patch_size,
                        :
                        ]
                patches.append(patch)

        return patches

    def get_pollution_level(self, concentration):
        """根据浓度确定污染等级"""
        if concentration < 500:
            return 0
        elif concentration < 1000:
            return 1
        elif concentration < 2000:
            return 2
        elif concentration < 3000:
            return 3
        else:
            return 4

    def create_example_labels(self):
        """创建示例标签文件"""
        labels_df = pd.DataFrame({
            'sample_id': range(1, 15),
            'concentration': [
                3402.02, 19836.94, 21663.57, 27298.00,
                12130.86, 8639.12, 21663.57, 18272.88,
                3073.18, 23111.74, 2098.58, 19718.26,
                270.36, 3402.02
            ]
        })

        labels_df.to_excel("data/raw/labels.xlsx", index=False)
        print("✅ 已创建示例标签文件")


if __name__ == "__main__":
    np.random.seed(42)
    processor = DataProcessor()
    processor.prepare_data()