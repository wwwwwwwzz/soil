"""
结果分析和可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import pickle

def analyze_results():
    # 加载模型和数据 - 修复：添加weights_only=False
    checkpoint = torch.load('../checkpoints/best_model_improved.pth', weights_only=False)
    print(f"最佳模型来自Epoch {checkpoint['epoch']+1}")
    print(f"验证R²: {checkpoint['val_r2']:.4f}")
    print(f"验证准确率: {checkpoint['val_acc']:.4f}")

    # 加载预处理参数
    try:
        with open('../data/processed/preprocessing_params.pkl', 'rb') as f:
            params = pickle.load(f)
            print(f"标签映射: {params.get('label_mapping', 'Not found')}")
    except:
        print("预处理参数文件未找到")
        params = {}

    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    # 创建可视化
    fig = plt.figure(figsize=(15, 10))

    # 1. 加载验证数据进行详细分析
    print("\n加载验证数据...")
    val_data = np.load('../data/processed/val_data.npy')
    val_labels = np.load('../data/processed/val_labels.npy')
    val_concentrations = np.load('../data/processed/val_concentrations.npy')

    print(f"验证数据形状: {val_data.shape}")
    print(f"标签唯一值: {np.unique(val_labels)}")
    print(f"浓度范围: {val_concentrations.min():.2f} - {val_concentrations.max():.2f} mg/kg")

    # 2. 浓度分布分析
    ax1 = plt.subplot(2, 3, 1)
    unique_labels = np.unique(val_labels)
    colors = ['blue', 'orange', 'green', 'red', 'purple']

    for i, label in enumerate(unique_labels):
        mask = val_labels == label
        ax1.hist(val_concentrations[mask], bins=30, alpha=0.6,
                label=f'Level {label} (n={mask.sum()})', color=colors[i])

    ax1.set_xlabel('Concentration (mg/kg)')
    ax1.set_ylabel('Count')
    ax1.set_title('Concentration Distribution by Pollution Level')
    ax1.legend()
    ax1.set_yscale('log')

    # 3. 箱线图
    ax2 = plt.subplot(2, 3, 2)
    data_for_box = []
    labels_for_box = []

    for label in unique_labels:
        mask = val_labels == label
        data_for_box.append(val_concentrations[mask])
        labels_for_box.append(f'Level {label}')

    bp = ax2.boxplot(data_for_box, labels=labels_for_box)
    ax2.set_ylabel('Concentration (mg/kg)')
    ax2.set_title('Concentration Boxplot by Level')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 4. 光谱方差分析
    ax3 = plt.subplot(2, 3, 3)
    # 计算各波段的方差
    data_reshaped = val_data.reshape(-1, val_data.shape[-1])
    band_variance = np.var(data_reshaped, axis=0)
    band_mean = np.mean(data_reshaped, axis=0)

    wavelengths = np.linspace(889, 1710, len(band_variance))

    ax3.plot(wavelengths, band_variance / band_variance.max(),
             'b-', label='Normalized Variance', alpha=0.7)
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Normalized Variance')
    ax3.set_title('Spectral Variance Analysis')
    ax3.grid(True, alpha=0.3)

    # 标记高方差区域
    high_var_idx = np.where(band_variance > np.percentile(band_variance, 90))[0]
    ax3.scatter(wavelengths[high_var_idx],
               (band_variance[high_var_idx] / band_variance.max()),
               color='red', s=10, alpha=0.5, label='High Variance Bands')
    ax3.legend()

    # 5. 平均光谱按等级
    ax4 = plt.subplot(2, 3, 4)
    for i, label in enumerate(unique_labels):
        mask = val_labels == label
        mean_spectrum = val_data[mask].reshape(-1, val_data.shape[-1]).mean(axis=0)
        ax4.plot(wavelengths, mean_spectrum, label=f'Level {label}',
                color=colors[i], linewidth=2)

    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Mean Spectra by Pollution Level')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 6. 类别分布饼图
    ax5 = plt.subplot(2, 3, 5)
    label_counts = [np.sum(val_labels == label) for label in unique_labels]
    ax5.pie(label_counts, labels=[f'Level {l}' for l in unique_labels],
            autopct='%1.1f%%', startangle=90, colors=colors[:len(unique_labels)])
    ax5.set_title('Sample Distribution')

    # 7. 关键波段标注
    ax6 = plt.subplot(2, 3, 6)
    mean_spectrum_all = data_reshaped.mean(axis=0)
    ax6.plot(wavelengths, mean_spectrum_all, 'b-', linewidth=2)

    # 标注碳氢化合物吸收峰
    hc_peaks = {
        889: 'CH4',
        1215: 'C-H',
        1400: 'O-H',
        1680: 'HC',
        1730: 'Main'
    }

    for wl, label in hc_peaks.items():
        if wl >= wavelengths.min() and wl <= wavelengths.max():
            idx = np.argmin(np.abs(wavelengths - wl))
            ax6.axvline(x=wl, color='r', linestyle='--', alpha=0.3)
            ax6.annotate(label, xy=(wl, mean_spectrum_all[idx]),
                        xytext=(wl+10, mean_spectrum_all[idx]+0.01),
                        fontsize=8, color='red',
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

    ax6.set_xlabel('Wavelength (nm)')
    ax6.set_ylabel('Intensity')
    ax6.set_title('Mean Spectrum with Key Absorption Bands')
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Hyperspectral Soil Oil Contamination Analysis', fontsize=16, y=1.02)
    plt.tight_layout()

    # 保存图片
    results_path = Path('../results/figures')
    results_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_path / 'detailed_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 统计分析
    print("\n" + "="*60)
    print("详细统计分析")
    print("="*60)

    # 各等级的样本数和统计
    for label in unique_labels:
        mask = val_labels == label
        count = mask.sum()
        conc_data = val_concentrations[mask]

        print(f"\n污染等级 {label}:")
        print(f"  样本数: {count} ({count/len(val_labels)*100:.1f}%)")
        print(f"  浓度统计:")
        print(f"    平均值: {conc_data.mean():.2f} mg/kg")
        print(f"    中位数: {np.median(conc_data):.2f} mg/kg")
        print(f"    标准差: {conc_data.std():.2f} mg/kg")
        print(f"    最小值: {conc_data.min():.2f} mg/kg")
        print(f"    最大值: {conc_data.max():.2f} mg/kg")
        print(f"    25%分位: {np.percentile(conc_data, 25):.2f} mg/kg")
        print(f"    75%分位: {np.percentile(conc_data, 75):.2f} mg/kg")

    # 总体统计
    print("\n" + "="*60)
    print("总体数据统计")
    print("="*60)
    print(f"总样本数: {len(val_labels)}")
    print(f"浓度范围: {val_concentrations.min():.2f} - {val_concentrations.max():.2f} mg/kg")
    print(f"浓度均值: {val_concentrations.mean():.2f} mg/kg")
    print(f"浓度中位数: {np.median(val_concentrations):.2f} mg/kg")

    # 计算不同浓度范围的样本分布
    print("\n浓度区间分布:")
    ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000),
              (5000, 10000), (10000, 20000), (20000, 30000)]

    for min_c, max_c in ranges:
        mask = (val_concentrations >= min_c) & (val_concentrations < max_c)
        count = mask.sum()
        if count > 0:
            print(f"  {min_c:5d}-{max_c:5d} mg/kg: {count:4d} 样本 ({count/len(val_concentrations)*100:5.1f}%)")

    # 模型性能总结
    print("\n" + "="*60)
    print("模型性能总结")
    print("="*60)
    print(f"最佳模型来自: Epoch {checkpoint['epoch']+1}")
    print(f"验证损失: {checkpoint['val_loss']:.4f}")
    print(f"分类准确率: {checkpoint['val_acc']:.4f}")
    print(f"浓度预测R²: {checkpoint['val_r2']:.4f}")

    return checkpoint

if __name__ == "__main__":
    checkpoint = analyze_results()
    print("\n分析完成！结果已保存到 results/figures/detailed_analysis.png")