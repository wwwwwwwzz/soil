"""
wwz 250921
Random Forest基线对比
使用PCA降维和优化参数加速训练
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from pathlib import Path
import time
import joblib
from tqdm import tqdm

def train_rf_baseline_optimized():
    print("="*60)
    print("Random Forest基线模型（优化版）")
    print("="*60)

    # 加载数据
    data_path = Path('../data/processed')

    print("\n加载数据...")
    train_data = np.load(data_path / 'train_data.npy')
    val_data = np.load(data_path / 'val_data.npy')

    # 方案1：使用平均光谱而不是展平整个patch
    print("\n数据处理策略：")
    print("1. 使用平均光谱（降维到512）")

    # 对每个10×10 patch计算平均光谱
    train_data_mean = train_data.mean(axis=(1, 2))  # (N, 512)
    val_data_mean = val_data.mean(axis=(1, 2))      # (N, 512)

    print(f"   训练数据: {train_data.shape} → {train_data_mean.shape}")
    print(f"   验证数据: {val_data.shape} → {val_data_mean.shape}")

    # 方案2：PCA进一步降维
    print("\n2. PCA降维（保留99%方差）")
    pca = PCA(n_components=0.99, random_state=42)
    train_data_pca = pca.fit_transform(train_data_mean)
    val_data_pca = pca.transform(val_data_mean)

    print(f"   PCA后维度: {train_data_pca.shape[1]}")
    print(f"   解释方差比: {pca.explained_variance_ratio_.sum():.4f}")

    # 加载标签
    train_labels = np.load(data_path / 'train_labels.npy')
    train_concentrations = np.load(data_path / 'train_concentrations.npy')
    val_labels = np.load(data_path / 'val_labels.npy')
    val_concentrations = np.load(data_path / 'val_concentrations.npy')

    # ========== 分类任务 ==========
    print("\n" + "="*40)
    print("分类任务")
    print("="*40)

    # 使用优化的参数
    rf_clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_split=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    start_time = time.time()
    print("训练分类器...")
    rf_clf.fit(train_data_pca, train_labels)
    clf_time = time.time() - start_time
    print(f"训练时间: {clf_time:.2f}秒")

    # 评估分类
    train_pred_cls = rf_clf.predict(train_data_pca)
    val_pred_cls = rf_clf.predict(val_data_pca)

    train_acc = accuracy_score(train_labels, train_pred_cls)
    val_acc = accuracy_score(val_labels, val_pred_cls)

    print(f"\n分类结果:")
    print(f"  训练准确率: {train_acc:.4f}")
    print(f"  验证准确率: {val_acc:.4f}")

    # ========== 回归任务 ==========
    print("\n" + "="*40)
    print("回归任务")
    print("="*40)

    # 对浓度进行对数变换（与Transformer一致）
    train_conc_log = np.log1p(train_concentrations)
    val_conc_log = np.log1p(val_concentrations)

    rf_reg = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_split=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    start_time = time.time()
    print("训练回归器（使用对数浓度）...")
    rf_reg.fit(train_data_pca, train_conc_log)
    reg_time = time.time() - start_time
    print(f"训练时间: {reg_time:.2f}秒")

    # 预测并反变换
    train_pred_log = rf_reg.predict(train_data_pca)
    val_pred_log = rf_reg.predict(val_data_pca)

    train_pred_reg = np.expm1(train_pred_log)
    val_pred_reg = np.expm1(val_pred_log)

    # 评估回归
    train_r2 = r2_score(train_concentrations, train_pred_reg)
    val_r2 = r2_score(val_concentrations, val_pred_reg)
    val_rmse = np.sqrt(mean_squared_error(val_concentrations, val_pred_reg))
    val_mae = mean_absolute_error(val_concentrations, val_pred_reg)

    print(f"\n回归结果:")
    print(f"  训练R²: {train_r2:.4f}")
    print(f"  验证R²: {val_r2:.4f}")
    print(f"  验证RMSE: {val_rmse:.2f} mg/kg")
    print(f"  验证MAE: {val_mae:.2f} mg/kg")

    # ========== 特征重要性分析 ==========
    print("\n" + "="*40)
    print("特征重要性分析")
    print("="*40)

    # 获取PCA前的特征重要性
    rf_clf_full = RandomForestClassifier(
        n_estimators=30,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_clf_full.fit(train_data_mean, train_labels)

    feature_importance = rf_clf_full.feature_importances_
    top_features = np.argsort(feature_importance)[-10:]

    wavelengths = np.linspace(889, 1710, 512)
    print("\nTop 10 重要波段:")
    for idx in top_features:
        print(f"  波长 {wavelengths[idx]:.1f}nm: 重要性 {feature_importance[idx]:.4f}")

    # ========== 性能对比 ==========
    print("\n" + "="*60)
    print("模型性能对比")
    print("="*60)
    print(f"{'指标':<20} {'Random Forest':<15} {'Transformer':<15} {'提升':<10}")
    print("-"*60)

    # Transformer的结果（根据具体结果修改）
    trans_acc = 0.9434
    trans_r2 = 0.9275
    trans_rmse = 2506.39
    trans_mae = 1769.23

    # 计算提升
    acc_improve = (trans_acc - val_acc) / val_acc * 100
    r2_improve = (trans_r2 - val_r2) / abs(val_r2) * 100 if val_r2 != 0 else 100
    rmse_improve = (val_rmse - trans_rmse) / val_rmse * 100

    print(f"{'分类准确率':<18} {val_acc:>14.4f} {trans_acc:>14.4f} {acc_improve:>9.1f}%")
    print(f"{'浓度R²':<18} {val_r2:>14.4f} {trans_r2:>14.4f} {r2_improve:>9.1f}%")
    print(f"{'RMSE (mg/kg)':<18} {val_rmse:>14.2f} {trans_rmse:>14.2f} {rmse_improve:>9.1f}%")
    print(f"{'MAE (mg/kg)':<18} {val_mae:>14.2f} {trans_mae:>14.2f} {''}")

    # 保存模型
    print("\n保存模型...")
    joblib.dump(rf_clf, '../checkpoints/rf_classifier_optimized.pkl')
    joblib.dump(rf_reg, '../checkpoints/rf_regressor_optimized.pkl')
    joblib.dump(pca, '../checkpoints/pca_transformer.pkl')

    print("模型已保存")

    # 额外分析：不同数据表示的对比
    print("\n" + "="*60)
    print("不同数据表示方法对比")
    print("="*60)

    results = {
        '方法': ['平均光谱+PCA', '全展平(51200维)', 'Transformer(原始3D)'],
        '维度': [train_data_pca.shape[1], 51200, '10×10×512'],
        '训练时间': [f'{clf_time+reg_time:.1f}s', '>>1000s', '~300s'],
        'R²': [val_r2, '未完成', trans_r2]
    }

    for key in results.keys():
        print(f"{key:<12}: {results[key]}")

    return val_acc, val_r2, val_rmse

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)

    # 运行本版本
    acc, r2, rmse = train_rf_baseline_optimized()

    print("\n" + "="*60)
    print("结论")
    print("="*60)
    print("1. Random Forest在高维数据上效率低下")
    print("2. 使用平均光谱+PCA可以大幅加速")
    print("3. Transformer能直接处理3D结构，性能更优")
    print(f"4. Transformer相对RF提升: R²提升{(0.9275-r2)/abs(r2)*100:.1f}%")