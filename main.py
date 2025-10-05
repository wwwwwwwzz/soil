"""
主程序：一键运行完整实验流程
使用方法：python main.py
"""

import os
import sys
from pathlib import Path

# 步骤1：数据准备
print("="*50)
print("步骤1：准备数据...")
from utils.preprocessing import DataProcessor
processor = DataProcessor()
processor.prepare_data()

# 步骤2：数据增强
print("="*50)
print("步骤2：数据增强...")
from utils.augmentation import DataAugmentor
augmentor = DataAugmentor()
augmentor.augment_all()

# 步骤3：训练模型
print("="*50)
print("步骤3：训练Transformer模型...")
from experiments.train import TrainingPipeline
trainer = TrainingPipeline()
trainer.run_training()

# 步骤4：交叉验证
print("="*50)
print("步骤4：5折交叉验证...")
from experiments.cross_validation import CrossValidator
cv = CrossValidator()
cv_results = cv.run_cv()

# 步骤5：SHAP分析
print("="*50)
print("步骤5：SHAP可解释性分析...")
from explainability.shap_analysis import SHAPAnalyzer
analyzer = SHAPAnalyzer()
analyzer.analyze()

# 步骤6：基线对比
print("="*50)
print("步骤6：与Random Forest等基线对比...")
from experiments.baseline_comparison import BaselineComparison
baseline = BaselineComparison()
baseline.compare_all()

# 步骤7：生成报告
print("="*50)
print("步骤7：生成最终报告...")
from utils.visualization import ReportGenerator
reporter = ReportGenerator()
reporter.generate_final_report()

print("="*50)
print("✅ 实验完成！结果保存在 results/ 文件夹")