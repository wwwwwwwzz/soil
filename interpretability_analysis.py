"""
interpretability_analysis.py
物理可解释性分析工具 - 改进版
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class PhysicsInterpretabilityAnalyzer:
    """物理可解释性分析器"""

    def __init__(self, model, wavelengths):
        self.model = model
        self.wavelengths = wavelengths
        self.device = next(model.parameters()).device

    def analyze_physics_consistency(self, val_loader, num_samples=100):
        """分析物理一致性"""
        self.model.eval()

        all_attentions = []
        all_predictions = []
        all_labels = []
        all_concentrations = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_samples // val_loader.batch_size:
                    break

                data, labels, norm_conc, orig_conc, _ = batch
                data = data.to(self.device)

                outputs = self.model(data, return_attention=True)
                class_logits, regression, attention = outputs

                all_attentions.append(attention.cpu().numpy())
                all_predictions.append(regression.cpu().numpy())
                all_labels.append(labels.numpy())
                all_concentrations.append(orig_conc.numpy())

        # 合并所有批次
        all_attentions = np.concatenate(all_attentions)
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        all_concentrations = np.concatenate(all_concentrations)

        return {
            'attentions': all_attentions,
            'predictions': all_predictions,
            'labels': all_labels,
            'concentrations': all_concentrations
        }

    def calculate_spectral_correlation(self, attention_weights, ref_weights):
        """计算光谱相关性"""
        # 归一化处理
        attention_norm = (attention_weights - np.min(attention_weights)) / (np.max(attention_weights) - np.min(attention_weights) + 1e-8)
        ref_norm = (ref_weights - np.min(ref_weights)) / (np.max(ref_weights) - np.min(ref_weights) + 1e-8)

        correlation, _ = pearsonr(attention_norm, ref_norm)
        return abs(correlation)

    def calculate_peak_alignment(self, attention_weights, ref_weights):
        """计算峰值对齐度 - 改进版"""
        # 归一化
        attention_norm = (attention_weights - np.min(attention_weights)) / (np.max(attention_weights) - np.min(attention_weights) + 1e-8)
        ref_norm = (ref_weights - np.min(ref_weights)) / (np.max(ref_weights) - np.min(ref_weights) + 1e-8)

        # 使用相对高度和合适的参数检测峰值
        peaks_model, props_model = find_peaks(
            attention_norm,
            height=np.max(attention_norm) * 0.2,  # 降低到20%最大值
            prominence=0.1,  # 峰的突出度
            distance=5  # 峰之间最小距离
        )

        peaks_ref, props_ref = find_peaks(
            ref_norm,
            height=np.max(ref_norm) * 0.2,
            prominence=0.1,
            distance=5
        )

        if len(peaks_model) == 0 and len(peaks_ref) == 0:
            return 1.0  # 都没有峰，认为是匹配的

        if len(peaks_model) == 0 or len(peaks_ref) == 0:
            return 0.0  # 一个有峰一个没有

        # 计算对齐度（允许一定误差）
        alignment_score = 0
        tolerance = 10  # 允许10个波段的误差（约16nm）

        matched_peaks = set()
        for peak_m in peaks_model:
            for peak_r in peaks_ref:
                if abs(peak_m - peak_r) <= tolerance and peak_r not in matched_peaks:
                    alignment_score += 1
                    matched_peaks.add(peak_r)
                    break

        # 计算对齐率
        alignment_rate = alignment_score / max(len(peaks_model), len(peaks_ref))

        # 额外奖励：如果主要峰（最高的3个峰）对齐
        if len(peaks_model) >= 3 and len(peaks_ref) >= 3:
            # 获取最高的3个峰
            top_model = peaks_model[np.argsort(props_model['peak_heights'])[-3:]]
            top_ref = peaks_ref[np.argsort(props_ref['peak_heights'])[-3:]]

            top_matches = 0
            for peak_m in top_model:
                for peak_r in top_ref:
                    if abs(peak_m - peak_r) <= tolerance:
                        top_matches += 1
                        break

            # 如果主要峰对齐良好，提升分数
            if top_matches >= 2:
                alignment_rate = min(1.0, alignment_rate * 1.2)

        return alignment_rate

    def calculate_beer_lambert_consistency(self, predictions, concentrations):
        """计算Beer-Lambert定律一致性"""
        # 取对数处理浓度（Beer-Lambert定律）
        log_conc = np.log1p(concentrations)

        # 计算相关性
        correlation = r2_score(log_conc, predictions.squeeze())
        return max(0, correlation)

    def plot_physics_analysis(self, results, save_path='physics_analysis.png'):
        """绘制物理分析图 - 改进版"""
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 注意力权重分布
        attention_mean = results['attentions'].mean()
        axes[0, 0].hist(results['attentions'].flatten(), bins=50, alpha=0.7, color='blue')
        axes[0, 0].axvline(attention_mean, color='red', linestyle='--', label=f'Mean: {attention_mean:.3f}')
        axes[0, 0].set_xlabel('Attention Weight')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Attention Weight Distribution')
        axes[0, 0].legend()

        # 2. 光谱注意力模式 - 改进可视化
        mean_attention = results['attentions'].mean(axis=0)
        ref_curve = self.model.physics_attention.absorption_curve.cpu().numpy()

        # 归一化用于显示
        mean_attention_norm = (mean_attention - mean_attention.min()) / (mean_attention.max() - mean_attention.min() + 1e-8)
        ref_curve_norm = (ref_curve - ref_curve.min()) / (ref_curve.max() - ref_curve.min() + 1e-8)

        wavelength_values = self.wavelengths.numpy()
        axes[0, 1].plot(wavelength_values, mean_attention_norm, 'b-', label='Model Attention', linewidth=2)
        axes[0, 1].plot(wavelength_values, ref_curve_norm, 'r--', label='HC Reference', linewidth=2, alpha=0.7)

        # 标记峰值
        peaks_model, _ = find_peaks(mean_attention_norm, height=0.2, prominence=0.1)
        peaks_ref, _ = find_peaks(ref_curve_norm, height=0.2, prominence=0.1)

        axes[0, 1].plot(wavelength_values[peaks_model], mean_attention_norm[peaks_model], 'bo', markersize=8, label=f'Model Peaks ({len(peaks_model)})')
        axes[0, 1].plot(wavelength_values[peaks_ref], ref_curve_norm[peaks_ref], 'r^', markersize=8, label=f'Ref Peaks ({len(peaks_ref)})')

        axes[0, 1].set_xlabel('Wavelength (nm)')
        axes[0, 1].set_ylabel('Normalized Weight')
        axes[0, 1].set_title('Spectral Attention Pattern with Peaks')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 浓度预测相关性
        axes[0, 2].scatter(results['concentrations'], results['predictions'], alpha=0.5)
        axes[0, 2].plot([results['concentrations'].min(), results['concentrations'].max()],
                       [results['concentrations'].min(), results['concentrations'].max()],
                       'r--', label='Perfect Prediction')
        axes[0, 2].set_xlabel('True Concentration (mg/kg)')
        axes[0, 2].set_ylabel('Predicted Value')
        axes[0, 2].set_title('Concentration Prediction')
        axes[0, 2].legend()

        # 4. Beer-Lambert定律验证
        log_conc = np.log1p(results['concentrations'])
        axes[1, 0].scatter(log_conc, results['predictions'], alpha=0.5, color='green')

        # 添加趋势线
        z = np.polyfit(log_conc, results['predictions'].squeeze(), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(log_conc.min(), log_conc.max(), 100)
        axes[1, 0].plot(x_trend, p(x_trend), 'r-', label=f'Trend (R²={r2_score(log_conc, results["predictions"].squeeze()):.3f})')

        axes[1, 0].set_xlabel('log(1 + Concentration)')
        axes[1, 0].set_ylabel('Model Output')
        axes[1, 0].set_title('Beer-Lambert Law Consistency')
        axes[1, 0].legend()

        # 5. 类别与浓度一致性
        for label in np.unique(results['labels']):
            mask = results['labels'] == label
            axes[1, 1].scatter(results['concentrations'][mask],
                             results['predictions'][mask],
                             alpha=0.5, label=f'Class {label}')
        axes[1, 1].set_xlabel('True Concentration (mg/kg)')
        axes[1, 1].set_ylabel('Predicted Value')
        axes[1, 1].set_title('Class-Concentration Consistency')
        axes[1, 1].legend()

        # 6. 物理一致性评分
        scores = {
            'Spectral Correlation': self.calculate_spectral_correlation(mean_attention, ref_curve),
            'Peak Alignment': self.calculate_peak_alignment(mean_attention, ref_curve),
            'Beer-Lambert': self.calculate_beer_lambert_consistency(results['predictions'], results['concentrations']),
            'Overall': 0
        }
        scores['Overall'] = np.mean([scores['Spectral Correlation'], scores['Peak Alignment'], scores['Beer-Lambert']])

        # 评分可视化
        score_names = list(scores.keys())
        score_values = list(scores.values())
        colors = ['green' if s > 0.7 else 'orange' if s > 0.5 else 'red' for s in score_values]

        bars = axes[1, 2].bar(range(len(score_names)), score_values, color=colors)
        axes[1, 2].set_xticks(range(len(score_names)))
        axes[1, 2].set_xticklabels(score_names, rotation=45, ha='right')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Physics Consistency Scores')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        axes[1, 2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)

        # 添加数值标签
        for bar, value in zip(bars, score_values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{value:.3f}', ha='center', va='bottom')

        plt.suptitle('Physics-Constrained Model Interpretability Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return scores


def generate_interpretability_report(model, val_loader, save_dir='results/'):
    """生成完整的可解释性报告 - 改进版"""
    from pathlib import Path
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    analyzer = PhysicsInterpretabilityAnalyzer(
        model,
        torch.linspace(889, 1710, 512)
    )

    # 分析物理一致性
    print("分析物理一致性...")
    results = analyzer.analyze_physics_consistency(val_loader)

    # 生成可视化
    print("生成可视化报告...")
    scores = analyzer.plot_physics_analysis(
        results,
        save_path=save_dir / 'physics_analysis.png'
    )

    # 生成文本报告
    with open(save_dir / 'physics_report.txt', 'w', encoding='utf-8') as f:
        f.write("Physics-Constrained Model Interpretability Report\n")
        f.write("=" * 50 + "\n\n")

        f.write("1. Physics Consistency Scores:\n")
        for metric, score in scores.items():
            status = "✓ Excellent" if score > 0.8 else "○ Good" if score > 0.6 else "△ Needs Improvement"
            f.write(f"   {metric}: {score:.4f} {status}\n")

        f.write("\n2. Key Findings:\n")
        f.write(f"   - Model attention correlation with HC reference: {scores['Spectral Correlation']:.3f}\n")
        f.write(f"   - Peak alignment accuracy: {scores['Peak Alignment']:.1%}\n")
        f.write(f"   - Beer-Lambert law consistency: {scores['Beer-Lambert']:.3f}\n")

        f.write("\n3. Interpretation:\n")
        if scores['Overall'] > 0.7:
            f.write("   The model successfully learned physically meaningful features.\n")
            f.write("   High correlation indicates the model focuses on known HC absorption bands.\n")
            f.write("   Peak alignment shows the model identifies critical spectral features.\n")
        elif scores['Overall'] > 0.5:
            f.write("   The model shows moderate physics consistency.\n")
            f.write("   Some improvement in peak alignment could enhance interpretability.\n")
        else:
            f.write("   The model needs stronger physics constraints.\n")
            f.write("   Consider adjusting the physics loss weight or absorption curve definition.\n")

        f.write("\n4. Recommendations:\n")
        if scores['Peak Alignment'] < 0.5:
            f.write("   - Increase physics constraint weight (gamma) in training\n")
            f.write("   - Add explicit peak alignment loss\n")
            f.write("   - Verify wavelength-band mapping\n")

        if scores['Beer-Lambert'] < 0.7:
            f.write("   - Strengthen Beer-Lambert constraint in loss function\n")
            f.write("   - Consider logarithmic transformation of concentrations\n")

        if scores['Spectral Correlation'] < 0.8:
            f.write("   - Review absorption curve definition\n")
            f.write("   - Check if model is learning correct spectral features\n")

    print(f"Interpretability report saved to {save_dir}")

    return scores