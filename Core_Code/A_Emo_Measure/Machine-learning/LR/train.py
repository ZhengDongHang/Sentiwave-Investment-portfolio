import os
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

from utils import load_all_data  # 🔄 使用封装好的读取逻辑

# ---------------- 参数配置 ----------------
folder_path = '../../data/train_data/'  # 训练集json文件夹路径
save_model_path = 'linear_model.pkl'    # 模型保存路径
max_len = 64                            # 最大文本长度

# ---------------- 加载 + 清洗数据 ----------------
print("🔄 加载训练数据...")
all_data = load_all_data(folder_path, if_train=True)

# ---------------- 生成训练样本 ----------------
print("🔨 处理训练样本...")
texts = []
targets = []

for item in tqdm(all_data, desc="构建样本"):
    index = item['index']
    for text in item.get('guba_data', []):
        if len(text) > max_len:
            text = text[:max_len]  # 截断文本
        texts.append(text)
        targets.append(index)

targets = np.array(targets)

# ---------------- 特征提取 ----------------
print("🔎 提取特征...")
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(texts)

X = X_text  # 仅使用文本特征，无需拼接来源

# ---------------- 模型训练 ----------------
print("🏋️ 正在训练模型...")
model = LinearRegression()
model.fit(X, targets)

# ---------------- 保存模型 ----------------
print("💾 保存模型到", save_model_path)
joblib.dump((vectorizer, model), save_model_path)

print("✅ 训练完成，模型已保存。")
