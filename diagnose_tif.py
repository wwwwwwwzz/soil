"""
诊断TIF文件格式
运行: python diagnose_tif.py
"""

import tifffile as tiff
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def diagnose_tif():
    """深入诊断TIF文件格式"""
    data_path = Path("data/raw/hyperspectral")

    # 只检查第一个文件
    tif_file = data_path / "sample_01.tif"

    if not tif_file.exists():
        print(f"文件不存在: {tif_file}")
        return

    print("=" * 60)
    print("TIF文件格式诊断")
    print("=" * 60)

    # 1. 读取文件
    img = tiff.imread(str(tif_file))
    print(f"\n文件: {tif_file.name}")
    print(f"原始形状: {img.shape}")
    print(f"数据类型: {img.dtype}")
    print(f"内存大小: {img.nbytes / 1024 / 1024:.2f} MB")

    # 2. 分析每个维度
    print(f"\n维度分析:")
    print(f"  第1维 (shape[0]): {img.shape[0]}")
    print(f"  第2维 (shape[1]): {img.shape[1]}")
    if img.ndim > 2:
        print(f"  第3维 (shape[2]): {img.shape[2]}")

    # 3. 尝试不同的解释方式
    print(f"\n可能的格式解释:")

    if img.ndim == 3:
        # 假设1: (bands, height, width)
        if img.shape[0] > 100 and img.shape[0] < 1000:
            print(f"  格式1: (bands, height, width)")
            print(f"    - 波段数: {img.shape[0]}")
            print(f"    - 图像尺寸: {img.shape[1]} × {img.shape[2]}")

            # 显示一个波段
            band_50 = img[50, :, :]
            print(f"    - 第50波段统计: min={band_50.min():.2f}, max={band_50.max():.2f}, mean={band_50.mean():.2f}")

        # 假设2: (height, width, bands)
        if img.shape[2] > 100 and img.shape[2] < 1000:
            print(f"  格式2: (height, width, bands)")
            print(f"    - 图像尺寸: {img.shape[0]} × {img.shape[1]}")
            print(f"    - 波段数: {img.shape[2]}")

            # 显示一个像素的光谱
            pixel_spectrum = img[100, 100, :]
            print(f"    - 像素(100,100)光谱统计: min={pixel_spectrum.min():.2f}, max={pixel_spectrum.max():.2f}")

    # 4. 检查是否是多页TIF
    print(f"\n检查多页TIF:")
    try:
        with tiff.TiffFile(str(tif_file)) as tif:
            print(f"  页数: {len(tif.pages)}")
            if len(tif.pages) > 1:
                print(f"  这是一个多页TIF文件！")
                for i, page in enumerate(tif.pages[:5]):  # 只显示前5页
                    print(f"    第{i}页: {page.shape}")
    except Exception as e:
        print(f"  检查失败: {e}")

    # 5. 可视化诊断
    print(f"\n创建诊断图...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 假设是 (bands, height, width)
    if img.shape[0] > 100:
        # 显示不同波段
        bands_to_show = [0, img.shape[0] // 4, img.shape[0] // 2, 3 * img.shape[0] // 4, img.shape[0] - 1]
        for i, band_idx in enumerate(bands_to_show[:3]):
            ax = axes[0, i]
            im = ax.imshow(img[band_idx, :, :], cmap='viridis')
            ax.set_title(f'波段 {band_idx}')
            plt.colorbar(im, ax=ax)

        # 显示一个像素的光谱曲线
        ax = axes[1, 0]
        center_x, center_y = img.shape[1] // 2, img.shape[2] // 2
        spectrum = img[:, center_x, center_y]
        ax.plot(spectrum)
        ax.set_title(f'像素({center_x},{center_y})的光谱')
        ax.set_xlabel('波段索引')
        ax.set_ylabel('强度')

        # 显示平均光谱
        ax = axes[1, 1]
        mean_spectrum = img.mean(axis=(1, 2))
        ax.plot(mean_spectrum)
        ax.set_title('平均光谱')
        ax.set_xlabel('波段索引')
        ax.set_ylabel('强度')

        # 显示光谱统计
        ax = axes[1, 2]
        ax.text(0.1, 0.8, f'假设格式: (bands, height, width)', fontsize=12)
        ax.text(0.1, 0.6, f'波段数: {img.shape[0]}', fontsize=12)
        ax.text(0.1, 0.4, f'空间尺寸: {img.shape[1]}×{img.shape[2]}', fontsize=12)
        ax.text(0.1, 0.2, f'波长范围: 889-1710nm (假设)', fontsize=12)
        ax.axis('off')

    # 假设是 (height, width, bands)
    else:
        print("似乎不是高光谱图像格式")

    plt.tight_layout()
    plt.savefig('tif_diagnosis.png', dpi=150)
    print(f"✅ 诊断图已保存为 tif_diagnosis.png")

    # 6. 检查所有文件的一致性
    print(f"\n检查所有文件的一致性:")
    all_shapes = []
    for i in range(1, 15):
        file_path = data_path / f"sample_{i:02d}.tif"
        if file_path.exists():
            img = tiff.imread(str(file_path))
            all_shapes.append(img.shape)
            print(f"  sample_{i:02d}.tif: {img.shape}")

    # 分析一致性
    first_dims = [s[0] for s in all_shapes]
    if len(set(first_dims)) == 1:
        print(f"\n✅ 所有文件第一个维度相同: {first_dims[0]} (很可能是波段数)")
    else:
        print(f"\n⚠️ 第一个维度不一致: {set(first_dims)}")


if __name__ == "__main__":
    diagnose_tif()