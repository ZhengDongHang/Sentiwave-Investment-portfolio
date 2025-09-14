import os
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor

from utils import load_all_data  # 你自己的数据加载函数

# -------------- 参数 --------------
folder_path = '../../data/train_data/'  # 数据路径
save_model_path = 'svr_model.pkl'
max_len = 64

# -------------- 加载数据 --------------
print("🔄 加载训练数据...")
all_data = load_all_data(folder_path, if_train=True)

# -------------- 构造样本 --------------
print("🔨 处理训练样本...")
texts = []
targets = []

for item in tqdm(all_data, desc="构建样本"):
    index = item['index']  # 三维连续标签，如 (x1,x2,x3)
    for text in item.get('guba_data', []):
        if len(text) > max_len:
            text = text[:max_len]
        texts.append(text)
        targets.append(index)

targets = np.array(targets)  # (N,3) 形状

# -------------- 特征提取 --------------
print("🔎 提取文本特征...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# -------------- 模型训练 --------------
print("🏋️ 训练多标签回归模型...")
base_model = LinearSVR()
model = MultiOutputRegressor(base_model)
model.fit(X, targets)

# -------------- 保存模型 --------------
print("💾 保存模型到", save_model_path)
joblib.dump((vectorizer, model), save_model_path)

print("✅ 训练完成。")
