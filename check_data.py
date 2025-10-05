"""
检查TIF数据格式
运行: python check_data.py
"""

import os
from pathlib import Path
import tifffile as tiff
import numpy as np


def check_tif_files():
    """检查TIF文件格式"""
    data_path = Path("data/raw/hyperspectral")

    if not data_path.exists():
        print(f"❌ 文件夹不存在: {data_path}")
        print("   请创建文件夹并放入TIF文件")
        return

    tif_files = list(data_path.glob("*.tif")) + list(data_path.glob("*.tiff"))

    if not tif_files:
        print(f"❌ 没有找到TIF文件在: {data_path}")
        return

    print(f"找到 {len(tif_files)} 个TIF文件\n")

    for i, file_path in enumerate(tif_files, 1):
        print(f"文件 {i}: {file_path.name}")

        try:
            # 读取TIF文件
            img = tiff.imread(str(file_path))

            print(f"  形状: {img.shape}")
            print(f"  数据类型: {img.dtype}")
            print(f"  值范围: [{img.min():.2f}, {img.max():.2f}]")

            # 判断数据格式
            if img.ndim == 2:
                print(f"  格式: 单波段图像")
            elif img.ndim == 3:
                if img.shape[0] < 50 and img.shape[1] > 50 and img.shape[2] > 50:
                    print(f"  格式: (bands, height, width)")
                    print(f"  波段数: {img.shape[0]}")
                elif img.shape[2] < 1000:
                    print(f"  格式: (height, width, bands)")
                    print(f"  波段数: {img.shape[2]}")
                else:
                    print(f"  格式: 未知")

            print()

        except Exception as e:
            print(f"  ❌ 读取失败: {e}\n")


if __name__ == "__main__":
    print("=" * 50)
    print("TIF文件格式检查")
    print("=" * 50)

    check_tif_files()

    print("\n提示：")
    print("1. 确保所有TIF文件命名为 sample_01.tif, sample_02.tif, ...")
    print("2. 创建labels.xlsx文件，包含concentration列")
    print("3. 运行 python utils/preprocessing.py 处理数据")