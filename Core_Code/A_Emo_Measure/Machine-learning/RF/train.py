import os
import joblib
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from utils import load_all_data  # 🔄 导入统一的数据加载模块

# ---------------- 参数配置 ----------------
folder_path     = '../../data/train_data/'   # JSON 文件夹路径
save_model_path = 'rf_model.pkl'             # 模型保存路径
max_features    = 5000                        # TF-IDF 最大特征数
n_estimators    = 50                          # 随机森林中树的数量
max_depth       = None                        # 最大深度，None 表示不限制
max_len         = 64                          # 最大文本长度

# ---------------- 1. 加载并组织数据 ----------------
print("📦 加载并清洗数据...")
data_list = load_all_data(folder_path, if_train=True)

texts = []
labels = []

for item in data_list:
    idx_vals = item.get("index", None)
    if idx_vals is None:
        continue

    for text in item.get("guba_data", []):
        if len(text) > max_len:
            text = text[:max_len]  # 截断文本
        texts.append(text)
        labels.append(idx_vals)

labels = np.array(labels)
print(f"✅ 样本数：{len(texts)}，标签维度：{labels.shape}")

# ---------------- 2. 特征提取 ----------------
print("🔍 提取 TF-IDF 特征...")
vectorizer = TfidfVectorizer(max_features=max_features)
X_train = vectorizer.fit_transform(texts)

# ---------------- 3. 随机森林训练 ----------------
print("🌲 训练随机森林模型...")
n_outputs = labels.shape[1]
estimators = []

for i in tqdm(range(n_outputs), desc="训练维度", unit="dim"):
    y_i = labels[:, i]
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_i)
    estimators.append(rf)

# ---------------- 4. 保存模型 ----------------
os.makedirs(os.path.dirname(save_model_path) or '.', exist_ok=True)
joblib.dump((vectorizer, estimators), save_model_path)
print(f"✅ 模型保存至 {save_model_path}")
