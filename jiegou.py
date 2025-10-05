"""
高光谱土壤污染检测Transformer网络结构图
科研论文风格，真实反映模型架构
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np


def create_network_diagram():
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))

    # 设置坐标系
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # 定义颜色方案（科研风格）
    colors = {
        'input': '#E8F4F8',
        'embedding': '#B8E0E8',
        'transformer': '#96C7D4',
        'attention': '#74AEC0',
        'ffn': '#5295AC',
        'output': '#FFE4B5',
        'pool': '#D4E6F1',
        'norm': '#E8DAEF'
    }

    # 垂直位置
    y_positions = {
        'input': 18,
        'spectral_embed': 16,
        'pos_embed': 15,
        'transformer1': 12,
        'transformer2': 8,
        'pool': 5,
        'output_cls': 2.5,
        'output_reg': 2.5
    }

    # 1. 输入层
    input_box = FancyBboxPatch(
        (3, y_positions['input']), 4, 1,
        boxstyle="round,pad=0.05",
        facecolor=colors['input'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(5, y_positions['input'] + 0.5, 'Input Hyperspectral Patches',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_positions['input'] + 0.2, '(B, 10, 10, 512)',
            ha='center', va='center', fontsize=9, style='italic')

    # 2. 光谱嵌入层
    embed_box = FancyBboxPatch(
        (3, y_positions['spectral_embed']), 4, 0.8,
        boxstyle="round,pad=0.05",
        facecolor=colors['embedding'],
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(embed_box)
    ax.text(5, y_positions['spectral_embed'] + 0.4, 'Spectral Embedding',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, y_positions['spectral_embed'] + 0.1, 'Linear: 512 → 128',
            ha='center', va='center', fontsize=8)

    # 位置编码（并行显示）
    pos_box = FancyBboxPatch(
        (7.5, y_positions['pos_embed']), 2, 0.6,
        boxstyle="round,pad=0.05",
        facecolor=colors['norm'],
        edgecolor='gray',
        linewidth=1,
        linestyle='--'
    )
    ax.add_patch(pos_box)
    ax.text(8.5, y_positions['pos_embed'] + 0.3, 'Positional',
            ha='center', va='center', fontsize=9)
    ax.text(8.5, y_positions['pos_embed'] + 0.1, 'Encoding',
            ha='center', va='center', fontsize=9)

    # 加号
    ax.text(6.8, y_positions['pos_embed'] + 0.3, '+',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # 3. Transformer Block 1
    def draw_transformer_block(y_base, block_num):
        # 主框架
        main_box = Rectangle((2, y_base), 6, 3,
                             facecolor=colors['transformer'],
                             edgecolor='black',
                             linewidth=2,
                             alpha=0.3)
        ax.add_patch(main_box)

        ax.text(5, y_base + 2.7, f'Transformer Block {block_num}',
                ha='center', va='center', fontsize=10, fontweight='bold')

        # Multi-Head Attention
        mha_box = FancyBboxPatch(
            (2.5, y_base + 1.8), 2.2, 0.6,
            boxstyle="round,pad=0.02",
            facecolor=colors['attention'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(mha_box)
        ax.text(3.6, y_base + 2.1, 'Multi-Head Attention',
                ha='center', va='center', fontsize=8)
        ax.text(3.6, y_base + 1.9, '(4 heads, d=128)',
                ha='center', va='center', fontsize=7, style='italic')

        # Layer Norm 1
        ln1_box = Rectangle((5, y_base + 1.85), 0.8, 0.5,
                            facecolor=colors['norm'],
                            edgecolor='gray',
                            linewidth=1)
        ax.add_patch(ln1_box)
        ax.text(5.4, y_base + 2.1, 'LN', ha='center', va='center', fontsize=8)

        # Add & Norm notation
        ax.text(6.1, y_base + 2.1, 'Add &', ha='center', va='center', fontsize=7)

        # FFN
        ffn_box = FancyBboxPatch(
            (2.5, y_base + 0.8), 2.2, 0.6,
            boxstyle="round,pad=0.02",
            facecolor=colors['ffn'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(ffn_box)
        ax.text(3.6, y_base + 1.1, 'Feed Forward',
                ha='center', va='center', fontsize=8)
        ax.text(3.6, y_base + 0.9, '128→512→128',
                ha='center', va='center', fontsize=7, style='italic')

        # Layer Norm 2
        ln2_box = Rectangle((5, y_base + 0.85), 0.8, 0.5,
                            facecolor=colors['norm'],
                            edgecolor='gray',
                            linewidth=1)
        ax.add_patch(ln2_box)
        ax.text(5.4, y_base + 1.1, 'LN', ha='center', va='center', fontsize=8)

        # Add & Norm notation
        ax.text(6.1, y_base + 1.1, 'Add &', ha='center', va='center', fontsize=7)

        # Dropout notation
        ax.text(7.2, y_base + 1.5, 'Dropout\n(p=0.3)',
                ha='center', va='center', fontsize=7, style='italic', color='red')

        # Residual connections (虚线)
        # Attention residual
        ax.plot([3.6, 3.6], [y_base + 1.8, y_base + 1.5], 'k--', alpha=0.5, linewidth=1)
        ax.plot([3.6, 6.5], [y_base + 1.5, y_base + 1.5], 'k--', alpha=0.5, linewidth=1)
        ax.plot([6.5, 6.5], [y_base + 1.5, y_base + 2.1], 'k--', alpha=0.5, linewidth=1)

        # FFN residual
        ax.plot([3.6, 3.6], [y_base + 0.8, y_base + 0.5], 'k--', alpha=0.5, linewidth=1)
        ax.plot([3.6, 6.5], [y_base + 0.5, y_base + 0.5], 'k--', alpha=0.5, linewidth=1)
        ax.plot([6.5, 6.5], [y_base + 0.5, y_base + 1.1], 'k--', alpha=0.5, linewidth=1)

    draw_transformer_block(y_positions['transformer1'], 1)
    draw_transformer_block(y_positions['transformer2'], 2)

    # 4. Global Average Pooling
    pool_box = FancyBboxPatch(
        (3, y_positions['pool']), 4, 0.8,
        boxstyle="round,pad=0.05",
        facecolor=colors['pool'],
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(pool_box)
    ax.text(5, y_positions['pool'] + 0.4, 'Global Average Pooling',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, y_positions['pool'] + 0.1, 'mean(dim=1) → (B, 128)',
            ha='center', va='center', fontsize=8, style='italic')

    # 5. 输出层（双任务）
    # 分类头
    cls_box = FancyBboxPatch(
        (1.5, y_positions['output_cls']), 2.5, 0.8,
        boxstyle="round,pad=0.05",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(cls_box)
    ax.text(2.75, y_positions['output_cls'] + 0.5, 'Classification Head',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(2.75, y_positions['output_cls'] + 0.2, 'Linear: 128 → 3',
            ha='center', va='center', fontsize=8)

    # 回归头
    reg_box = FancyBboxPatch(
        (6, y_positions['output_reg']), 2.5, 0.8,
        boxstyle="round,pad=0.05",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(reg_box)
    ax.text(7.25, y_positions['output_reg'] + 0.5, 'Regression Head',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(7.25, y_positions['output_reg'] + 0.2, 'Linear: 128 → 1',
            ha='center', va='center', fontsize=8)

    # 输出标签
    ax.text(2.75, 1.5, 'Pollution Level\n(0, 3, 4)',
            ha='center', va='center', fontsize=8, style='italic')
    ax.text(7.25, 1.5, 'Concentration\n(mg/kg)',
            ha='center', va='center', fontsize=8, style='italic')

    # 6. 连接箭头
    arrow_props = dict(arrowstyle='->', lw=2, color='black')

    # 输入到嵌入
    ax.annotate('', xy=(5, y_positions['spectral_embed'] + 0.8),
                xytext=(5, y_positions['input']),
                arrowprops=arrow_props)

    # 嵌入到Transformer 1
    ax.annotate('', xy=(5, y_positions['transformer1'] + 3),
                xytext=(5, y_positions['pos_embed']),
                arrowprops=arrow_props)

    # 位置编码连接
    ax.annotate('', xy=(7, y_positions['pos_embed'] + 0.3),
                xytext=(7.5, y_positions['pos_embed'] + 0.3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', linestyle='--'))

    # Transformer 1 到 Transformer 2
    ax.annotate('', xy=(5, y_positions['transformer2'] + 3),
                xytext=(5, y_positions['transformer1']),
                arrowprops=arrow_props)

    # Transformer 2 到池化
    ax.annotate('', xy=(5, y_positions['pool'] + 0.8),
                xytext=(5, y_positions['transformer2']),
                arrowprops=arrow_props)

    # 池化到输出（分叉）
    ax.annotate('', xy=(2.75, y_positions['output_cls'] + 0.8),
                xytext=(4, y_positions['pool']),
                arrowprops=arrow_props)
    ax.annotate('', xy=(7.25, y_positions['output_reg'] + 0.8),
                xytext=(6, y_positions['pool']),
                arrowprops=arrow_props)

    # 添加标题
    ax.text(5, 19.5, 'Spectral-Spatial Transformer for Soil Oil Contamination Detection',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # 添加参数标注
    param_text = "Model Parameters: 0.14M\nInput: 512 bands (889-1710nm)\nPatch Size: 10×10 pixels"
    ax.text(9.5, 18, param_text,
            ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray'))

    # 添加损失函数标注
    loss_text = "Multi-Task Loss:\nL = 0.4×L_cls + 0.6×L_reg\nL_cls: CrossEntropy\nL_reg: MSE"
    ax.text(0.5, 4, loss_text,
            ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFF9E6', edgecolor='gray'))

    plt.tight_layout()

    # 保存图片
    plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('network_architecture.pdf', format='pdf', bbox_inches='tight', facecolor='white')

    plt.show()

    print("网络结构图已保存为:")
    print("  - network_architecture.png (300 DPI)")
    print("  - network_architecture.pdf (矢量图)")


if __name__ == "__main__":
    create_network_diagram()