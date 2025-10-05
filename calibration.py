"""
高光谱数据白板校准和暗场校准脚本
用于处理土壤高光谱数据的辐射定标
"""

import os
import numpy as np
import spectral.io.envi as envi
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import json
from datetime import datetime
import logging

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperspectralCalibrator:
    """高光谱数据校准器"""

    def __init__(self, soil_data_path: str, output_path: str):
        """
        初始化校准器

        Args:
            soil_data_path: 土壤数据文件夹路径
            output_path: 输出文件夹路径
        """
        self.soil_data_path = Path(soil_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 创建子文件夹
        self.calibrated_path = self.output_path / 'calibrated'
        self.calibrated_path.mkdir(exist_ok=True)

        self.white_reference = None
        self.dark_reference = None
        self.calibration_params = {}

        # 曝光时间参数
        self.sample_exposure_time = 200000  # 土壤样本曝光时间（微秒）
        self.calibration_exposure_time = 100000  # 校准数据曝光时间（微秒）
        self.exposure_correction_factor = None  # 曝光时间校正系数

    def read_envi_file(self, file_prefix: str) -> np.ndarray:
        """
        读取ENVI格式文件

        Args:
            file_prefix: 文件前缀（不包含扩展名）

        Returns:
            numpy数组格式的高光谱数据
        """
        hdr_file = self.soil_data_path / f"{file_prefix}.hdr"
        spe_file = self.soil_data_path / f"{file_prefix}.spe"

        if not hdr_file.exists() or not spe_file.exists():
            raise FileNotFoundError(f"无法找到文件: {file_prefix}")

        try:
            # 读取ENVI文件
            img = envi.open(str(hdr_file), str(spe_file))
            data = img.load()
            logger.info(f"成功读取文件: {file_prefix}, 数据维度: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"读取文件 {file_prefix} 失败: {str(e)}")
            raise

    def select_calibration_lines(self, data: np.ndarray,
                                white_line_idx: Optional[int] = None,
                                dark_line_idx: Optional[int] = None,
                                white_line_range: Optional[int] = 1,
                                dark_line_range: Optional[int] = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        从校准文件中选择白板和暗场参考线

        Args:
            data: 校准文件数据
            white_line_idx: 白板线中心索引（如果为None，则交互式选择）
            dark_line_idx: 暗场线中心索引（如果为None，则交互式选择）
            white_line_range: 白板线的范围（从中心线向上下各取几行进行平均）
            dark_line_range: 暗场线的范围（从中心线向上下各取几行进行平均）

        Returns:
            白板参考数据和暗场参考数据 (width, bands) - 保留每个像素点的独立校准值
        """
        height, width, bands = data.shape

        if white_line_idx is None or dark_line_idx is None:
            # 显示平均光谱强度图像用于选择
            mean_intensity = np.mean(data, axis=2)

            plt.figure(figsize=(12, 8))
            plt.imshow(mean_intensity, cmap='gray', aspect='auto')
            plt.colorbar(label='平均强度')
            plt.title('选择白板线（亮）和暗场线（暗）')
            plt.xlabel('列（相机线阵方向）')
            plt.ylabel('行（扫描方向）')

            # 显示一些候选线的平均强度
            fig, axes = plt.subplots(2, 1, figsize=(12, 6))

            # 显示每行的平均强度
            row_means = np.mean(mean_intensity, axis=1)
            axes[0].plot(row_means)
            axes[0].set_xlabel('行索引（扫描方向）')
            axes[0].set_ylabel('平均强度')
            axes[0].set_title('每行的平均强度（用于选择校准线）')
            axes[0].grid(True, alpha=0.3)

            # 找出最亮和最暗的几行作为候选
            bright_rows = np.argsort(row_means)[-10:]  # 最亮的10行
            dark_rows = np.argsort(row_means)[:10]      # 最暗的10行

            axes[1].bar(range(len(row_means)), row_means, color='gray', alpha=0.5)
            axes[1].bar(bright_rows, row_means[bright_rows], color='yellow', label='候选白板线')
            axes[1].bar(dark_rows, row_means[dark_rows], color='blue', label='候选暗场线')
            axes[1].set_xlabel('行索引（扫描方向）')
            axes[1].set_ylabel('平均强度')
            axes[1].set_title('候选校准线')
            axes[1].legend()

            plt.tight_layout()
            plt.show()

            # 自动选择或手动输入
            logger.info(f"最亮的行索引（候选白板）: {bright_rows}")
            logger.info(f"最暗的行索引（候选暗场）: {dark_rows}")

            if white_line_idx is None:
                white_line_idx = int(input(f"请输入白板线的中心行索引 (建议: {bright_rows[-1]}): ") or bright_rows[-1])
            if dark_line_idx is None:
                dark_line_idx = int(input(f"请输入暗场线的中心行索引 (建议: {dark_rows[0]}): ") or dark_rows[0])

        # 询问是否要使用多行平均
        use_multi_lines = input("是否使用多行平均以提高信噪比？(y/n，默认n): ").lower() == 'y'
        if use_multi_lines:
            white_line_range = int(input(f"白板线范围（上下各取几行，默认3）: ") or 3)
            dark_line_range = int(input(f"暗场线范围（上下各取几行，默认3）: ") or 3)
        else:
            white_line_range = 0
            dark_line_range = 0

        # 提取选定的线（可能是多行）
        white_start = max(0, white_line_idx - white_line_range)
        white_end = min(height, white_line_idx + white_line_range + 1)
        dark_start = max(0, dark_line_idx - dark_line_range)
        dark_end = min(height, dark_line_idx + dark_line_range + 1)

        # 提取参考数据并沿着扫描方向（行）平均，但保留线阵方向（列）的各个像素
        white_reference = np.mean(data[white_start:white_end, :, :], axis=0)  # (width, bands)
        dark_reference = np.mean(data[dark_start:dark_end, :, :], axis=0)    # (width, bands)

        logger.info(f"白板参考线: 中心行{white_line_idx}, 范围[{white_start}:{white_end}]")
        logger.info(f"暗场参考线: 中心行{dark_line_idx}, 范围[{dark_start}:{dark_end}]")
        logger.info(f"白板参考数据形状: {white_reference.shape} (宽度×波段)")
        logger.info(f"暗场参考数据形状: {dark_reference.shape} (宽度×波段)")
        logger.info(f"白板强度范围: [{np.min(white_reference):.2f}, {np.max(white_reference):.2f}]")
        logger.info(f"暗场强度范围: [{np.min(dark_reference):.2f}, {np.max(dark_reference):.2f}]")

        # 保存校准参数
        self.calibration_params = {
            'white_line_idx': int(white_line_idx),
            'white_line_range': int(white_line_range),
            'dark_line_idx': int(dark_line_idx),
            'dark_line_range': int(dark_line_range),
            'white_intensity_range': [float(np.min(white_reference)), float(np.max(white_reference))],
            'dark_intensity_range': [float(np.min(dark_reference)), float(np.max(dark_reference))],
            'calibration_date': datetime.now().isoformat()
        }

        return white_reference, dark_reference

    def load_calibration_data(self, calibration_file: str = "12",
                            white_line_idx: Optional[int] = None,
                            dark_line_idx: Optional[int] = None,
                            sample_exposure: Optional[float] = None,
                            calibration_exposure: Optional[float] = None):
        """
        加载校准数据

        Args:
            calibration_file: 包含白板和暗场数据的文件名
            white_line_idx: 白板线索引
            dark_line_idx: 暗场线索引
            sample_exposure: 样本数据的曝光时间（微秒）
            calibration_exposure: 校准数据的曝光时间（微秒）
        """
        logger.info(f"正在加载校准文件: {calibration_file}")

        # 设置曝光时间
        if sample_exposure is not None:
            self.sample_exposure_time = sample_exposure
        if calibration_exposure is not None:
            self.calibration_exposure_time = calibration_exposure

        # 计算曝光时间校正系数
        self.exposure_correction_factor = self.sample_exposure_time / self.calibration_exposure_time
        logger.info(f"曝光时间校正: 样本 {self.sample_exposure_time}μs, 校准 {self.calibration_exposure_time}μs")
        logger.info(f"曝光校正系数: {self.exposure_correction_factor:.2f}")

        # 读取校准文件
        calibration_data = self.read_envi_file(calibration_file)

        # 选择校准线
        self.white_reference, self.dark_reference = self.select_calibration_lines(
            calibration_data, white_line_idx, dark_line_idx
        )

        # 应用曝光时间校正到白板参考
        # 白板参考需要根据曝光时间比例进行缩放
        # 公式：White_corrected = White_measured * (Exposure_sample / Exposure_calibration)
        logger.info("应用曝光时间校正到白板参考...")
        self.white_reference = self.white_reference * self.exposure_correction_factor

        # 暗场参考也需要相应调整（暗电流与曝光时间成正比）
        self.dark_reference = self.dark_reference * self.exposure_correction_factor

        logger.info(f"校正后白板强度范围: [{np.min(self.white_reference):.2f}, {np.max(self.white_reference):.2f}]")
        logger.info(f"校正后暗场强度范围: [{np.min(self.dark_reference):.2f}, {np.max(self.dark_reference):.2f}]")

        # 更新校准参数
        self.calibration_params['sample_exposure_time'] = self.sample_exposure_time
        self.calibration_params['calibration_exposure_time'] = self.calibration_exposure_time
        self.calibration_params['exposure_correction_factor'] = self.exposure_correction_factor

        # 可视化校准光谱
        self.visualize_calibration_spectra()

    def visualize_calibration_spectra(self):
        """可视化白板和暗场参考光谱（显示线上不同位置的光谱）"""
        if self.white_reference is None or self.dark_reference is None:
            logger.warning("校准数据未加载")
            return

        width, bands = self.white_reference.shape

        # 创建两个子图
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # 子图1: 显示几个代表性位置的光谱曲线
        band_indices = np.arange(bands)
        positions_to_show = [0, width//4, width//2, 3*width//4, width-1]  # 显示5个位置

        for pos in positions_to_show:
            axes[0].plot(band_indices, self.white_reference[pos, :],
                        label=f'白板-位置{pos}', alpha=0.7, linewidth=1)
            axes[0].plot(band_indices, self.dark_reference[pos, :],
                        label=f'暗场-位置{pos}', alpha=0.7, linewidth=1, linestyle='--')

        axes[0].set_xlabel('波段索引')
        axes[0].set_ylabel('强度值')
        axes[0].set_title('不同空间位置的校准参考光谱')
        axes[0].legend(loc='best', ncol=2)
        axes[0].grid(True, alpha=0.3)

        # 子图2: 显示校准参考的2D图像
        # 合并白板和暗场参考用于显示
        combined_ref = np.vstack([self.white_reference.T,
                                  np.zeros((20, width)),  # 间隔
                                  self.dark_reference.T])

        im = axes[1].imshow(combined_ref, aspect='auto', cmap='hot')
        axes[1].set_xlabel('像素位置（线阵方向）')
        axes[1].set_ylabel('波段索引 / 参考类型')
        axes[1].set_title('校准参考2D图（上：白板，下：暗场）')

        # 添加分割线和标注
        axes[1].axhline(y=bands, color='cyan', linewidth=2, linestyle='-')
        axes[1].axhline(y=bands+20, color='cyan', linewidth=2, linestyle='-')
        axes[1].text(5, bands//2, '白板参考', color='white', fontsize=10, va='center')
        axes[1].text(5, bands+20+bands//2, '暗场参考', color='white', fontsize=10, va='center')

        plt.colorbar(im, ax=axes[1], label='强度值')

        plt.tight_layout()

        # 保存图像
        plt.savefig(self.output_path / 'calibration_spectra_perpixel.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 输出统计信息
        logger.info(f"白板参考统计 - 空间变化系数: {np.std(np.mean(self.white_reference, axis=1))/np.mean(self.white_reference):.4f}")
        logger.info(f"暗场参考统计 - 空间变化系数: {np.std(np.mean(self.dark_reference, axis=1))/np.mean(self.dark_reference):.4f}")

    def calibrate_data(self, data: np.ndarray, apply_exposure_correction: bool = True) -> np.ndarray:
        """
        对数据进行白板和暗场校准（逐像素校准）

        公式: R = (DN - DC) / (WR - DC)
        其中:
        - R: 反射率
        - DN: 原始数据
        - DC: 暗场参考（每个像素独立，已经过曝光校正）
        - WR: 白板参考（每个像素独立，已经过曝光校正）

        Args:
            data: 原始高光谱数据 (height, width, bands)
            apply_exposure_correction: 是否应用曝光时间校正（默认True）

        Returns:
            校准后的反射率数据
        """
        if self.white_reference is None or self.dark_reference is None:
            raise ValueError("校准数据未加载，请先调用load_calibration_data()")

        # 校准参考数据维度: (width, bands)
        ref_width, ref_bands = self.white_reference.shape

        # 检查数据维度
        if len(data.shape) != 3:
            raise ValueError(f"期望3维数据 (height, width, bands)，但得到 {data.shape}")

        height, width, bands = data.shape

        # 检查宽度是否匹配
        if width != ref_width:
            logger.warning(f"数据宽度 {width} 与校准参考宽度 {ref_width} 不匹配")
            # 如果宽度不匹配，可能需要裁剪或插值
            if width > ref_width:
                # 数据宽度大于参考宽度，裁剪数据
                data = data[:, :ref_width, :]
                width = ref_width
            else:
                # 数据宽度小于参考宽度，裁剪参考
                white_ref = self.white_reference[:width, :]
                dark_ref = self.dark_reference[:width, :]
                ref_width = width
        else:
            white_ref = self.white_reference
            dark_ref = self.dark_reference

        # 检查波段数是否匹配
        if bands != ref_bands:
            logger.warning(f"数据波段数 {bands} 与校准波段数 {ref_bands} 不匹配")
            min_bands = min(bands, ref_bands)
            data = data[:, :, :min_bands]
            white_ref = white_ref[:, :min_bands]
            dark_ref = dark_ref[:, :min_bands]

        # 扩展参考数据到与输入数据相同的高度
        # white_ref 和 dark_ref: (width, bands) -> (1, width, bands) -> (height, width, bands)
        white_ref_expanded = np.expand_dims(white_ref, axis=0)
        dark_ref_expanded = np.expand_dims(dark_ref, axis=0)

        # 广播到完整尺寸
        white_ref_expanded = np.broadcast_to(white_ref_expanded, data.shape)
        dark_ref_expanded = np.broadcast_to(dark_ref_expanded, data.shape)

        # 执行逐像素校准
        denominator = white_ref_expanded - dark_ref_expanded

        # 避免除零
        denominator = np.where(denominator == 0, 1e-10, denominator)

        # 计算反射率
        reflectance = (data - dark_ref_expanded) / denominator

        # 限制范围在[0, 1.5] - 允许轻微超过1.0的反射率（某些材料可能有这种情况）
        reflectance = np.clip(reflectance, 0, 1.5)

        # 输出一些统计信息用于调试
        logger.debug(f"校准统计 - 最小反射率: {np.min(reflectance):.4f}, "
                    f"最大反射率: {np.max(reflectance):.4f}, "
                    f"平均反射率: {np.mean(reflectance):.4f}")

        # 检查是否有异常值
        if np.max(reflectance) > 1.2:
            logger.warning(f"检测到高反射率值 (>1.2): 最大值={np.max(reflectance):.4f}")
            logger.warning("这可能是由于：1) 曝光时间差异 2) 白板参考选择不当 3) 样本比白板更亮")

        return reflectance

    def process_single_file(self, file_prefix: str, save_as_envi: bool = True) -> np.ndarray:
        """
        处理单个文件

        Args:
            file_prefix: 文件前缀
            save_as_envi: 是否保存为ENVI格式

        Returns:
            校准后的数据
        """
        logger.info(f"正在处理文件: {file_prefix}")

        try:
            # 读取原始数据和头文件信息
            hdr_file = self.soil_data_path / f"{file_prefix}.hdr"
            spe_file = self.soil_data_path / f"{file_prefix}.spe"

            # 读取原始ENVI文件
            orig_img = envi.open(str(hdr_file), str(spe_file))
            raw_data = orig_img.load()

            # 执行校准
            calibrated_data = self.calibrate_data(raw_data)

            # 统计信息
            logger.info(f"原始数据维度: {raw_data.shape}")
            logger.info(f"校准前数据范围: [{np.min(raw_data):.2f}, {np.max(raw_data):.2f}]")
            logger.info(f"校准后数据维度: {calibrated_data.shape}")
            logger.info(f"校准后数据范围: [{np.min(calibrated_data):.4f}, {np.max(calibrated_data):.4f}]")

            # 保存校准后的数据
            if save_as_envi:
                output_hdr = self.calibrated_path / f"{file_prefix}_calibrated.hdr"
                output_spe = self.calibrated_path / f"{file_prefix}_calibrated.spe"

                # 复制原始元数据并更新必要字段
                metadata = orig_img.metadata.copy()

                # 更新数据类型为float32
                metadata['data type'] = 4  # ENVI的float32类型代码

                # 添加校准信息到描述中
                original_desc = metadata.get('description', '')
                metadata['description'] = f'{original_desc} - Calibrated with white/dark reference'

                # 确保数据格式正确
                # 转换数据布局以匹配原始文件的interleave格式
                interleave = metadata.get('interleave', 'bip').lower()

                if interleave == 'bsq':  # Band Sequential
                    # 数据需要转换为 (bands, lines, samples)
                    save_data = np.transpose(calibrated_data, (2, 0, 1))
                elif interleave == 'bil':  # Band Interleaved by Line
                    # 数据需要转换为 (lines, bands, samples)
                    save_data = np.transpose(calibrated_data, (0, 2, 1))
                else:  # bip - Band Interleaved by Pixel (默认)
                    # 数据保持原样 (lines, samples, bands)
                    save_data = calibrated_data

                # 保存为ENVI格式
                # 直接写入二进制文件
                save_data_float32 = save_data.astype(np.float32)
                save_data_float32.tofile(str(output_spe))

                # 写入头文件
                envi.write_envi_header(str(output_hdr), metadata)

                logger.info(f"已保存校准数据到: {output_spe}")
                logger.info(f"数据格式: {interleave}, 保存形状: {save_data.shape}")

            return calibrated_data

        except Exception as e:
            logger.error(f"处理文件 {file_prefix} 失败: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return None

    def process_all_files(self, file_list: list = None, use_second_collection: bool = False):
        """
        批量处理所有文件

        Args:
            file_list: 要处理的文件列表，如果为None则处理1-11
            use_second_collection: 是否使用第二次采集的数据（.1后缀）
        """
        if file_list is None:
            file_list = [str(i) for i in range(1, 12)]

        # 如果使用第二次采集的数据，添加.1后缀
        if use_second_collection:
            file_list = [f"{f}.1" for f in file_list]

        results = {}

        for file_prefix in file_list:
            calibrated_data = self.process_single_file(file_prefix)
            if calibrated_data is not None:
                results[file_prefix] = {
                    'shape': calibrated_data.shape,
                    'min': float(np.min(calibrated_data)),
                    'max': float(np.max(calibrated_data)),
                    'mean': float(np.mean(calibrated_data)),
                    'std': float(np.std(calibrated_data))
                }

        # 保存处理结果摘要
        summary_file = self.output_path / 'calibration_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'calibration_params': self.calibration_params,
                'processed_files': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        logger.info(f"校准完成！处理了 {len(results)} 个文件")
        logger.info(f"结果摘要已保存到: {summary_file}")

        return results

    def compare_before_after_interactive(self, file_prefix: str):
        """
        交互式比较校准前后的光谱

        Args:
            file_prefix: 文件前缀
        """
        # 不强制切换后端，使用当前后端
        import matplotlib.pyplot as plt

        # 读取原始数据
        print(f"正在读取文件 {file_prefix} 用于交互式分析...")
        raw_data = self.read_envi_file(file_prefix)

        # 确保数据是三维的
        if len(raw_data.shape) != 3:
            logger.error(f"数据维度错误: {raw_data.shape}，期望三维数据")
            return

        # 执行校准
        calibrated_data = self.calibrate_data(raw_data)

        height, width, bands = raw_data.shape
        print(f"数据维度: 高度={height}, 宽度={width}, 波段数={bands}")

        # 创建交互式图形
        fig = plt.figure(figsize=(15, 10))

        # 布局：2x2网格
        # 左上：原始数据的平均强度图
        ax1 = plt.subplot(2, 2, 1)
        raw_mean = np.mean(raw_data, axis=2)
        im1 = ax1.imshow(raw_mean, cmap='gray', aspect='auto', interpolation='nearest')
        ax1.set_title('原始数据（点击选择像素）')
        ax1.set_xlabel('列（x）')
        ax1.set_ylabel('行（y）')
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # 右上：校准后数据的平均强度图
        ax2 = plt.subplot(2, 2, 2)
        cal_mean = np.mean(calibrated_data, axis=2)
        im2 = ax2.imshow(cal_mean, cmap='gray', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
        ax2.set_title('校准后数据（反射率）')
        ax2.set_xlabel('列（x）')
        ax2.set_ylabel('行（y）')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        # 左下：光谱对比图
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_xlabel('波段索引')
        ax3.set_ylabel('原始强度值', color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('点击上方图像查看光谱')

        # 创建第二个y轴用于反射率
        ax4 = ax3.twinx()
        ax4.set_ylabel('反射率', color='g')
        ax4.tick_params(axis='y', labelcolor='g')

        # 右下：多点光谱显示
        ax5 = plt.subplot(2, 2, 4)
        ax5.set_xlabel('波段索引')
        ax5.set_ylabel('反射率')
        ax5.set_title('多点光谱对比（最多10个点）')
        ax5.grid(True, alpha=0.3)

        # 存储选中的点
        self.selected_points = []
        self.scatter_points = []

        # 添加文本提示
        info_text = fig.text(0.5, 0.02, '提示：左键点击选择像素 | C键清除所有点 | 点击窗口关闭按钮退出',
                            ha='center', fontsize=10, color='red')

        def on_click(event):
            """鼠标点击事件处理"""
            # 调试信息
            if event.inaxes:
                print(f"鼠标点击: button={event.button}, x={event.xdata:.1f}, y={event.ydata:.1f}")

            # 检查是否在图像区域内点击
            if event.inaxes in [ax1, ax2] and event.button == 1:  # 左键点击
                if event.xdata is not None and event.ydata is not None:
                    x, y = int(event.xdata), int(event.ydata)

                    # 确保坐标在有效范围内
                    if 0 <= x < width and 0 <= y < height:
                        print(f"选中像素: ({x}, {y})")

                        # 获取该点的光谱 - 确保正确提取一维数组
                        raw_spectrum = raw_data[y, x, :].flatten()  # 强制转换为一维
                        cal_spectrum = calibrated_data[y, x, :].flatten()  # 强制转换为一维

                        # 验证维度
                        print(f"光谱数据维度: raw={raw_spectrum.shape}, cal={cal_spectrum.shape}")

                        # 判断是否为有效数据点（排除背景）
                        mean_intensity = np.mean(raw_spectrum)
                        is_background = mean_intensity < np.percentile(raw_mean, 10)  # 低于10%分位数可能是背景

                        # 清除左下图的旧数据
                        ax3.clear()
                        ax4.clear()

                        # 绘制当前点的光谱
                        bands_idx = np.arange(bands)
                        line1 = ax3.plot(bands_idx, raw_spectrum, 'b-', linewidth=1.5,
                                        label=f'原始 ({x},{y})')[0]
                        line2 = ax4.plot(bands_idx, cal_spectrum, 'g-', linewidth=1.5,
                                        label=f'校准 ({x},{y})')[0]

                        # 如果反射率异常高，用红色虚线标记
                        if np.max(cal_spectrum) > 1.0:
                            ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='反射率=1.0')
                            ax3.set_title(f'像素 ({x},{y}) - 注意：反射率>1.0 {"(可能是背景)" if is_background else ""}',
                                        color='red' if is_background else 'black')
                        else:
                            ax3.set_title(f'像素 ({x},{y}) 的光谱')

                        ax3.set_xlabel('波段索引')
                        ax3.set_ylabel('原始强度值', color='b')
                        ax3.tick_params(axis='y', labelcolor='b')
                        ax3.grid(True, alpha=0.3)
                        ax3.legend(loc='upper left')

                        ax4.set_ylabel('反射率', color='g')
                        ax4.tick_params(axis='y', labelcolor='g')
                        ax4.legend(loc='upper right')

                        # 添加到多点对比图
                        if len(self.selected_points) < 10:  # 最多显示10个点
                            self.selected_points.append((x, y))

                            # 根据数据类型选择颜色
                            if is_background:
                                color = 'gray'
                                marker = 'x'
                            else:
                                color = plt.cm.tab10(len([p for p in self.selected_points
                                                         if np.mean(raw_data[p[1], p[0], :]) >= np.percentile(raw_mean, 10)]) - 1)
                                marker = 'o'

                            # 在右下图中添加新光谱
                            ax5.plot(bands_idx, cal_spectrum,
                                   color=color, alpha=0.7,
                                   label=f'({x},{y}){"[背景]" if is_background else ""}',
                                   linewidth=1, linestyle='--' if is_background else '-')

                            # 在图像上标记选中的点
                            for ax in [ax1, ax2]:
                                point = ax.scatter(x, y, c=[color], s=100,
                                                 edgecolors='white', linewidth=2,
                                                 marker=marker, alpha=0.8)
                                self.scatter_points.append(point)
                        else:
                            info_text.set_text('已达到最大点数(10个)，按C键清除后可继续选择')

                        ax5.legend(loc='best', fontsize=8, ncol=2)

                        # 显示统计信息
                        print(f"像素 ({x},{y}): 原始均值={mean_intensity:.2f}, "
                              f"校准后均值={np.mean(cal_spectrum):.4f}, "
                              f"最大反射率={np.max(cal_spectrum):.4f}"
                              f"{' [可能是背景]' if is_background else ''}")

                        # 刷新图形
                        plt.draw()

        def on_key(event):
            """键盘事件处理"""
            print(f"按键: {event.key}")

            if event.key in ['c', 'C']:  # 清除所有选中的点
                self.selected_points.clear()

                # 移除所有标记点
                for point in self.scatter_points:
                    point.remove()
                self.scatter_points.clear()

                # 清除多点光谱图
                ax5.clear()
                ax5.set_xlabel('波段索引')
                ax5.set_ylabel('反射率')
                ax5.set_title('多点光谱对比（最多10个点）')
                ax5.grid(True, alpha=0.3)

                info_text.set_text('提示：左键点击选择像素 | C键清除所有点 | 点击窗口关闭按钮退出')
                plt.draw()
                print("已清除所有选中的点")

        # 连接事件处理器
        cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
        cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

        plt.suptitle(f'文件 {file_prefix} - 交互式校准对比', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # 保存静态图像
        comparison_file = self.output_path / f'comparison_{file_prefix}_interactive.png'
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        logger.info(f"交互式对比图已保存到: {comparison_file}")

        # 显示交互式窗口
        plt.show()

        # 断开事件连接
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_key)

        print("\n交互完成！")


def main():
    """主函数"""
    # 配置路径
    SOIL_DATA_PATH = r"E:\project\turang\土壤数据250926"
    PROJECT_PATH = r"E:\project\soil_contamination_detection"
    OUTPUT_PATH = os.path.join(PROJECT_PATH, "data", "calibrated")

    # 创建校准器实例
    calibrator = HyperspectralCalibrator(SOIL_DATA_PATH, OUTPUT_PATH)

    print("=" * 60)
    print("高光谱数据校准程序")
    print("=" * 60)

    # 选择使用哪次采集的数据
    use_second = input("是否使用第二次采集的数据？(y/n，默认n): ").lower() == 'y'

    # 设置曝光时间参数
    print("\n曝光时间设置：")
    print("默认值：土壤样本 200000μs，校准数据 100000μs")
    use_custom = input("是否使用自定义曝光时间？(y/n，默认n): ").lower() == 'y'

    if use_custom:
        sample_exp = float(input("输入土壤样本曝光时间（微秒）: "))
        calib_exp = float(input("输入校准数据曝光时间（微秒）: "))
    else:
        sample_exp = 200000  # 土壤样本曝光时间
        calib_exp = 100000   # 校准数据曝光时间

    # 加载校准数据
    calibration_file = "12.1" if use_second else "12"
    print(f"\n正在加载校准文件: {calibration_file}")
    print("系统将显示强度图，请根据图像选择白板线和暗场线的行索引")

    # 加载校准数据并应用曝光时间校正
    calibrator.load_calibration_data(
        calibration_file=calibration_file,
        white_line_idx=None,  # 设置为None以交互式选择
        dark_line_idx=None,    # 设置为None以交互式选择
        sample_exposure=sample_exp,
        calibration_exposure=calib_exp
    )

    # 处理所有文件
    print("\n开始批量处理文件...")
    file_list = [str(i) for i in range(1, 12)]  # 处理1-11号文件
    results = calibrator.process_all_files(file_list, use_second_collection=use_second)

    # 显示处理结果
    print("\n处理结果摘要:")
    print("-" * 40)
    print(f"曝光时间校正系数: {calibrator.exposure_correction_factor:.2f}")
    print("-" * 40)
    for file_name, stats in results.items():
        print(f"{file_name}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
        if stats['max'] > 1.0:
            print(f"  ⚠️ 警告：最大反射率 {stats['max']:.4f} > 1.0")

    # 交互式对比校准前后的效果
    sample_file = "1.1" if use_second else "1"
    print(f"\n生成文件 {sample_file} 的交互式校准前后对比图...")
    print("您可以点击图像选择像素查看光谱，按'c'键清除所有选中的点")
    calibrator.compare_before_after_interactive(sample_file)

    print("\n校准完成！")
    print(f"校准后的数据保存在: {OUTPUT_PATH}")
    print(f"请查看 calibration_summary.json 了解详细信息")

    # 询问是否要查看其他文件
    while True:
        another = input("\n是否要查看其他文件的校准对比？(输入文件编号1-11，或n退出): ")
        if another.lower() == 'n':
            break
        try:
            file_num = int(another)
            if 1 <= file_num <= 11:
                file_to_compare = f"{file_num}.1" if use_second else str(file_num)
                calibrator.compare_before_after_interactive(file_to_compare)
            else:
                print("请输入1-11之间的数字")
        except ValueError:
            print("无效输入，请输入数字或'n'")

    print("\n程序结束，感谢使用！")


if __name__ == "__main__":
    main()