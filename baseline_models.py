"""
wwz
baseline_models.py
所有基线方法的实现
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')


class BaselineModel:
    """基线模型基类"""

    def __init__(self, model_type='plsr'):
        self.model_type = model_type
        self.classifier = None
        self.regressor = None
        self.fitted = False

    def fit(self, X_train, y_class, y_reg):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def save_model(self, path):
        import joblib
        joblib.dump({
            'model_type': self.model_type,
            'model_state': self.get_model_state()
        }, path)

    def load_model(self, path):
        import joblib
        data = joblib.load(path)
        self.model_type = data['model_type']
        self.set_model_state(data['model_state'])
        self.fitted = True

    def get_model_state(self):
        """获取模型状态用于保存"""
        return {}

    def set_model_state(self, state):
        """从保存的状态恢复模型"""
        pass


class PLSRBaseline(BaselineModel):
    """偏最小二乘回归基线"""

    def __init__(self, n_components=20):
        super().__init__(model_type='plsr')
        self.n_components = n_components
        self.pls = None

    def fit(self, X_train, y_class, y_reg):
        print(f"训练PLSR模型 (n_components={self.n_components})...")

        # 将空间维度展平
        X_flat = X_train.reshape(X_train.shape[0], -1)

        # PLSR同时处理分类和回归
        self.pls = PLSRegression(n_components=self.n_components)

        # 合并标签
        y_combined = np.column_stack([y_class, y_reg])
        self.pls.fit(X_flat, y_combined)

        self.fitted = True

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        y_pred = self.pls.predict(X_flat)

        # 分类预测
        class_scores = np.zeros((X.shape[0], 3))
        class_pred = np.round(y_pred[:, 0]).astype(int)
        class_pred = np.clip(class_pred, 0, 2)

        for i, c in enumerate(class_pred):
            class_scores[i, c] = 1.0

        # 回归预测
        reg_pred = y_pred[:, 1:2]

        return class_scores, reg_pred

    def get_model_state(self):
        """获取模型状态"""
        return {
            'n_components': self.n_components,
            'pls': self.pls
        }

    def set_model_state(self, state):
        """设置模型状态"""
        self.n_components = state['n_components']
        self.pls = state['pls']


class SVMBaseline(BaselineModel):
    """支持向量机基线"""

    def __init__(self, kernel='rbf', C=1.0):
        super().__init__(model_type='svm')
        self.kernel = kernel
        self.C = C
        self.svm_classifier = None
        self.svm_regressor = None

    def fit(self, X_train, y_class, y_reg):
        print(f"训练SVM模型 (kernel={self.kernel}, C={self.C})...")

        X_flat = X_train.reshape(X_train.shape[0], -1)

        # 分类SVM
        self.svm_classifier = SVC(kernel=self.kernel, C=self.C, probability=True)
        self.svm_classifier.fit(X_flat, y_class)

        # 回归SVM
        self.svm_regressor = SVR(kernel=self.kernel, C=self.C)
        self.svm_regressor.fit(X_flat, y_reg)

        self.fitted = True

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)

        class_proba = self.svm_classifier.predict_proba(X_flat)

        # 处理类别数不匹配
        if class_proba.shape[1] < 3:
            # 填充缺失的类别
            temp = np.zeros((X_flat.shape[0], 3))
            temp[:, :class_proba.shape[1]] = class_proba
            class_proba = temp

        reg_pred = self.svm_regressor.predict(X_flat).reshape(-1, 1)

        return class_proba, reg_pred

    def get_model_state(self):
        """获取模型状态"""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'svm_classifier': self.svm_classifier,
            'svm_regressor': self.svm_regressor
        }

    def set_model_state(self, state):
        """设置模型状态"""
        self.kernel = state['kernel']
        self.C = state['C']
        self.svm_classifier = state['svm_classifier']
        self.svm_regressor = state['svm_regressor']


class RandomForestBaseline(BaselineModel):
    """随机森林基线"""

    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__(model_type='rf')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.rf_classifier = None
        self.rf_regressor = None

    def fit(self, X_train, y_class, y_reg):
        print(f"训练随机森林模型 (n_estimators={self.n_estimators})...")

        X_flat = X_train.reshape(X_train.shape[0], -1)

        self.rf_classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.rf_classifier.fit(X_flat, y_class)

        self.rf_regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.rf_regressor.fit(X_flat, y_reg)

        self.fitted = True

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)

        class_proba = self.rf_classifier.predict_proba(X_flat)
        reg_pred = self.rf_regressor.predict(X_flat).reshape(-1, 1)

        return class_proba, reg_pred

    def get_model_state(self):
        """获取模型状态"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'rf_classifier': self.rf_classifier,
            'rf_regressor': self.rf_regressor
        }

    def set_model_state(self, state):
        """设置模型状态"""
        self.n_estimators = state['n_estimators']
        self.max_depth = state['max_depth']
        self.rf_classifier = state['rf_classifier']
        self.rf_regressor = state['rf_regressor']


class CNN1DBaseline(nn.Module):
    """1D-CNN基线（处理光谱维度）"""

    def __init__(self, num_bands=512, num_classes=3):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        # 计算展平后的尺寸
        self.flatten_size = 128 * (num_bands // 8)

        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)

        self.classifier = nn.Linear(128, num_classes)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, H, W, C) -> 取平均光谱
        x = x.mean(dim=(1, 2))  # (B, C)
        x = x.unsqueeze(1)  # (B, 1, C)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        features = torch.relu(self.fc2(x))

        class_logits = self.classifier(features)
        regression = self.regressor(features)

        return class_logits, regression


class CNN3DBaseline(nn.Module):
    """3D-CNN基线（空谱联合）"""

    def __init__(self, num_bands=512, num_classes=3):
        super().__init__()

        # 3D卷积层
        self.conv3d_1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 7), padding=(1, 1, 3))
        self.bn3d_1 = nn.BatchNorm3d(16)
        self.pool3d_1 = nn.MaxPool3d(kernel_size=(1, 1, 2))

        self.conv3d_2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 5), padding=(1, 1, 2))
        self.bn3d_2 = nn.BatchNorm3d(32)
        self.pool3d_2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # 2D卷积层（空间特征）
        self.conv2d_1 = nn.Conv2d(32 * (num_bands // 4), 64, kernel_size=3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(64)
        self.pool2d_1 = nn.MaxPool2d(2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)

        self.classifier = nn.Linear(128, num_classes)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, H, W, C) -> (B, 1, H, W, C)
        x = x.unsqueeze(1)

        # 3D卷积
        x = torch.relu(self.bn3d_1(self.conv3d_1(x)))
        x = self.pool3d_1(x)

        x = torch.relu(self.bn3d_2(self.conv3d_2(x)))
        x = self.pool3d_2(x)

        # 重排为2D: (B, C*D, H, W)
        B, C, H, W, D = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(B, C*D, H, W)

        # 2D卷积
        x = torch.relu(self.bn2d_1(self.conv2d_1(x)))
        x = self.pool2d_1(x)

        # 全连接
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        features = torch.relu(self.fc2(x))

        class_logits = self.classifier(features)
        regression = self.regressor(features)

        return class_logits, regression


class StandardTransformer(nn.Module):
    """标准Transformer基线（无物理约束）"""

    def __init__(self, num_bands=512, d_model=128, n_heads=4, n_layers=2,
                 num_classes=3, dropout=0.3):
        super().__init__()

        self.spectral_embedding = nn.Linear(num_bands, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(d_model, num_classes)
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape

        # 展平空间维度
        x = x.view(B, H * W, C)  # (B, H*W, C)

        # 光谱嵌入
        x = self.spectral_embedding(x)

        # 位置编码
        x = x + self.pos_embedding[:, :x.size(1), :]

        # Transformer编码
        x = self.transformer(x)

        # 全局池化
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)

        class_logits = self.classifier(x)
        regression = self.regressor(x)

        return class_logits, regression