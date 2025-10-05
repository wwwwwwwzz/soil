"""
高光谱图像ROI裁剪工具
用于框选和裁剪高光谱数据中的感兴趣区域（如培养皿）
"""

import os
import numpy as np
import spectral.io.envi as envi
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
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


class HyperspectralROICropper:
    """高光谱图像ROI裁剪器"""

    def __init__(self):
        """初始化裁剪器"""
        self.data = None
        self.metadata = None
        self.roi_list = []
        self.current_file = None
        self.fig = None
        self.ax = None
        self.selector = None

    def read_envi_file(self, file_path: str) -> tuple:
        """
        读取ENVI格式文件

        Args:
            file_path: 文件路径（不含扩展名）

        Returns:
            (data, metadata) 元组
        """
        # 构建完整路径
        file_path = Path(file_path)

        # 尝试不同的扩展名组合
        extensions = [('.hdr', '.spe'), ('.hdr', '.img'), ('.hdr', '.dat')]

        for hdr_ext, data_ext in extensions:
            hdr_file = file_path.parent / (file_path.stem + hdr_ext)
            data_file = file_path.parent / (file_path.stem + data_ext)

            if hdr_file.exists() and data_file.exists():
                try:
                    img = envi.open(str(hdr_file), str(data_file))
                    data = img.load()
                    metadata = img.metadata
                    logger.info(f"成功读取文件: {file_path.stem}")
                    logger.info(f"数据维度: {data.shape}")
                    return data, metadata
                except Exception as e:
                    continue

        raise FileNotFoundError(f"无法找到有效的ENVI文件: {file_path}")

    def save_roi(self, roi_data: np.ndarray, metadata: dict, output_path: str, roi_info: dict):
        """
        保存ROI数据为ENVI格式

        Args:
            roi_data: 裁剪后的数据
            metadata: 原始元数据
            output_path: 输出路径（不含扩展名）
            roi_info: ROI信息
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 更新元数据
        new_metadata = metadata.copy()
        new_metadata['lines'] = roi_data.shape[0]
        new_metadata['samples'] = roi_data.shape[1]
        new_metadata['bands'] = roi_data.shape[2] if len(roi_data.shape) == 3 else 1

        # 添加ROI信息到描述
        original_desc = metadata.get('description', '')
        new_metadata['description'] = (f'{original_desc}\nROI: x={roi_info["x"]}, y={roi_info["y"]}, '
                                       f'width={roi_info["width"]}, height={roi_info["height"]}')

        # 保存数据
        hdr_file = str(output_path) + '.hdr'
        data_file = str(output_path) + '.spe'

        # 根据原始数据格式保存
        interleave = metadata.get('interleave', 'bip').lower()

        if interleave == 'bsq':
            save_data = np.transpose(roi_data, (2, 0, 1))
        elif interleave == 'bil':
            save_data = np.transpose(roi_data, (0, 2, 1))
        else:  # bip
            save_data = roi_data

        # 保存为正确的数据类型
        dtype_map = {
            1: np.uint8, 2: np.int16, 3: np.int32,
            4: np.float32, 5: np.float64, 12: np.uint16
        }
        data_type = metadata.get('data type', 4)
        save_dtype = dtype_map.get(data_type, np.float32)

        save_data.astype(save_dtype).tofile(data_file)
        envi.write_envi_header(hdr_file, new_metadata)

        logger.info(f"ROI已保存到: {data_file}")

    def interactive_crop(self, input_file: str):
        """
        交互式裁剪界面

        Args:
            input_file: 输入文件路径（不含扩展名）
        """
        # 读取数据
        self.data, self.metadata = self.read_envi_file(input_file)
        self.current_file = Path(input_file).stem
        self.roi_list = []

        if len(self.data.shape) != 3:
            logger.error(f"数据维度错误: {self.data.shape}，需要三维数据")
            return

        height, width, bands = self.data.shape

        # 计算用于显示的平均强度图像
        display_img = np.mean(self.data, axis=2)

        # 创建交互式窗口
        self.fig, self.ax = plt.subplots(figsize=(12, 10))

        # 显示图像
        self.im = self.ax.imshow(display_img, cmap='gray', aspect='auto')
        self.ax.set_title(f'文件: {self.current_file} - 框选ROI区域（左键拖动）')
        self.ax.set_xlabel('列（x）')
        self.ax.set_ylabel('行（y）')
        plt.colorbar(self.im, ax=self.ax, fraction=0.046)

        # 添加操作说明
        instructions = (
            "操作说明:\n"
            "1. 左键拖动框选ROI\n"
            "2. 按S键保存当前ROI\n"
            "3. 按C键清除所有ROI\n"
            "4. 按A键保存所有ROI\n"
            "5. 按Q键退出"
        )
        self.fig.text(0.02, 0.98, instructions, transform=self.fig.transFigure,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 状态文本
        self.status_text = self.fig.text(0.5, 0.02, '就绪', ha='center',
                                         transform=self.fig.transFigure,
                                         fontsize=10, color='green')

        # 存储矩形补丁
        self.rect_patches = []

        def on_select(eclick, erelease):
            """矩形选择回调"""
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)

            # 确保坐标有效
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            # 确保x2 > x1, y2 > y1
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # 存储当前ROI
            self.current_roi = {
                'x': x1, 'y': y1,
                'width': x2 - x1, 'height': y2 - y1,
                'x2': x2, 'y2': y2
            }

            # 显示选择信息
            roi_area = self.current_roi['width'] * self.current_roi['height']
            self.status_text.set_text(
                f"当前选择: ({x1}, {y1}) 到 ({x2}, {y2}), "
                f"大小: {self.current_roi['width']}×{self.current_roi['height']}, "
                f"面积: {roi_area}像素"
            )

            print(f"框选区域: x={x1}-{x2}, y={y1}-{y2}")

        def on_key(event):
            """键盘事件处理"""
            if event.key == 's' or event.key == 'S':
                # 保存当前ROI
                if hasattr(self, 'current_roi'):
                    # 询问输出文件名
                    roi_num = len(self.roi_list) + 1
                    default_name = f"{self.current_file}_roi{roi_num}"

                    print(f"\n保存ROI #{roi_num}")
                    print(f"区域: x={self.current_roi['x']}-{self.current_roi['x2']}, "
                          f"y={self.current_roi['y']}-{self.current_roi['y2']}")

                    # 这里简化处理，直接使用默认名称
                    self.roi_list.append({
                        'roi_info': self.current_roi.copy(),
                        'output_name': default_name
                    })

                    # 在图像上绘制已保存的ROI
                    rect = patches.Rectangle(
                        (self.current_roi['x'], self.current_roi['y']),
                        self.current_roi['width'], self.current_roi['height'],
                        linewidth=2, edgecolor='lime', facecolor='none',
                        label=f'ROI {roi_num}'
                    )
                    self.ax.add_patch(rect)
                    self.rect_patches.append(rect)

                    # 添加标签
                    self.ax.text(self.current_roi['x'], self.current_roi['y'] - 5,
                                 f'ROI {roi_num}', color='lime', fontsize=10)

                    self.status_text.set_text(f"ROI #{roi_num} 已添加到列表")
                    plt.draw()
                else:
                    self.status_text.set_text("请先框选一个区域")

            elif event.key == 'c' or event.key == 'C':
                # 清除所有ROI
                self.roi_list = []
                for rect in self.rect_patches:
                    rect.remove()
                self.rect_patches = []
                self.status_text.set_text("已清除所有ROI")
                plt.draw()

            elif event.key == 'a' or event.key == 'A':
                # 保存所有ROI到文件
                if self.roi_list:
                    self.save_all_rois()
                else:
                    self.status_text.set_text("没有ROI需要保存")

            elif event.key == 'q' or event.key == 'Q':
                plt.close(self.fig)

        # 创建矩形选择器
        self.selector = RectangleSelector(
            self.ax, on_select,
            useblit=True,
            button=[1],  # 左键
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

        # 连接键盘事件
        self.fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()

    def save_all_rois(self):
        """保存所有ROI到文件"""
        # 询问输出文件夹
        output_dir = input("\n请输入输出文件夹路径（默认: ./cropped_rois）: ")
        if not output_dir:
            output_dir = "./cropped_rois"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存每个ROI
        for i, roi_item in enumerate(self.roi_list, 1):
            roi_info = roi_item['roi_info']
            output_name = roi_item['output_name']

            # 裁剪数据
            x1, y1 = roi_info['x'], roi_info['y']
            x2, y2 = roi_info['x2'], roi_info['y2']

            roi_data = self.data[y1:y2 + 1, x1:x2 + 1, :]

            # 保存
            output_path = output_dir / output_name
            self.save_roi(roi_data, self.metadata, str(output_path), roi_info)

            print(f"保存ROI {i}/{len(self.roi_list)}: {output_name}")

        # 保存ROI信息到JSON
        info_file = output_dir / f"{self.current_file}_roi_info.json"
        roi_info_list = []
        for roi_item in self.roi_list:
            info = roi_item['roi_info'].copy()
            info['output_name'] = roi_item['output_name']
            roi_info_list.append(info)

        with open(info_file, 'w') as f:
            json.dump({
                'source_file': self.current_file,
                'timestamp': datetime.now().isoformat(),
                'roi_list': roi_info_list
            }, f, indent=2)

        print(f"\n所有ROI已保存到: {output_dir}")
        print(f"ROI信息已保存到: {info_file}")
        self.status_text.set_text(f"已保存 {len(self.roi_list)} 个ROI到 {output_dir}")

    def batch_crop_with_template(self, template_roi_file: str, input_files: list, output_dir: str):
        """
        使用模板ROI批量裁剪多个文件

        Args:
            template_roi_file: 模板ROI信息JSON文件
            input_files: 输入文件列表
            output_dir: 输出目录
        """
        # 读取模板ROI
        with open(template_roi_file, 'r') as f:
            template_info = json.load(f)

        roi_list = template_info['roi_list']

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 批量处理每个文件
        for input_file in input_files:
            try:
                print(f"\n处理文件: {input_file}")
                data, metadata = self.read_envi_file(input_file)
                file_stem = Path(input_file).stem

                # 应用每个ROI
                for roi_info in roi_list:
                    x1, y1 = roi_info['x'], roi_info['y']
                    width, height = roi_info['width'], roi_info['height']
                    x2 = x1 + width
                    y2 = y1 + height

                    # 裁剪数据
                    roi_data = data[y1:y2, x1:x2, :]

                    # 生成输出名称
                    roi_name = roi_info.get('output_name', 'roi')
                    # 替换原始文件名
                    roi_name = roi_name.replace(template_info['source_file'], file_stem)

                    # 保存
                    output_path = output_dir / roi_name
                    self.save_roi(roi_data, metadata, str(output_path), roi_info)

            except Exception as e:
                logger.error(f"处理文件 {input_file} 失败: {str(e)}")
                continue

        print(f"\n批量裁剪完成！输出目录: {output_dir}")


def main():
    """主函数"""
    cropper = HyperspectralROICropper()

    print("=" * 60)
    print("高光谱图像ROI裁剪工具")
    print("=" * 60)

    while True:
        print("\n选择操作模式:")
        print("1. 交互式裁剪（手动框选）")
        print("2. 批量裁剪（使用模板）")
        print("3. 退出")

        choice = input("请选择 (1/2/3): ")

        if choice == '1':
            # 交互式裁剪
            input_file = input("\n请输入文件路径（不含扩展名）: ")
            if not input_file:
                print("示例: E:/project/turang/土壤数据250926/1")
                continue

            try:
                cropper.interactive_crop(input_file)
            except Exception as e:
                logger.error(f"处理失败: {str(e)}")

        elif choice == '2':
            # 批量裁剪
            template_file = input("\n请输入模板ROI信息文件（.json）: ")
            if not Path(template_file).exists():
                print("模板文件不存在")
                continue

            # 输入文件列表
            print("请输入要处理的文件（每行一个，输入空行结束）:")
            input_files = []
            while True:
                file_path = input().strip()
                if not file_path:
                    break
                input_files.append(file_path)

            if not input_files:
                print("没有输入文件")
                continue

            output_dir = input("请输入输出目录: ")
            if not output_dir:
                output_dir = "./batch_cropped"

            try:
                cropper.batch_crop_with_template(template_file, input_files, output_dir)
            except Exception as e:
                logger.error(f"批量处理失败: {str(e)}")

        elif choice == '3':
            print("程序退出")
            break
        else:
            print("无效选择")


if __name__ == "__main__":
    main()