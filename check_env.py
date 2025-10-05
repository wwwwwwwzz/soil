"""
环境检查脚本
运行：python check_env.py
"""

import sys
print("Python版本:", sys.version)

# 检查关键包
packages = {
    'numpy': '数据处理',
    'pandas': '数据框架',
    'torch': '深度学习',
    'matplotlib': '可视化',
    'sklearn': '机器学习',
    'shap': 'SHAP分析'
}

print("\n检查已安装的包：")
for pkg, desc in packages.items():
    try:
        if pkg == 'sklearn':
            import sklearn
            print(f"✅ {desc} (scikit-learn {sklearn.__version__})")
        else:
            module = __import__(pkg)
            print(f"✅ {desc} ({pkg} {module.__version__})")
    except ImportError:
        print(f"❌ {desc} ({pkg} 未安装)")

# 检查CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"\n✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA版本: {torch.version.cuda}")
    else:
        print("\n⚠️ CUDA不可用，将使用CPU运行")
except:
    pass

print("\n如果有❌标记，请安装对应的包")