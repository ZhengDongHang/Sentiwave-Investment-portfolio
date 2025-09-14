import os
import csv
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="Converting data to scipy sparse matrix")

# ---------------- 配置参数 ----------------
data_folder   = '../../data/test_data/'  # 测试集路径：多个个股 CSV 文件
model_path    = 'rf_model.pkl'           # 训练好的模型路径
output_folder = 'result/'                # 每个个股的预测结果输出路径
os.makedirs(output_folder, exist_ok=True)

# ---------------- 加载模型 ----------------
print("📦 加载模型...")
vectorizer, estimators = joblib.load(model_path)

# ---------------- 读取所有 CSV 文件 ----------------
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
print(f"📂 共发现 {len(csv_files)} 个 CSV 文件")

# ---------------- 遍历每只个股 ----------------
for file_name in tqdm(csv_files, desc="📊 正在处理"):
    input_path = os.path.join(data_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    try:
        df = pd.read_csv(input_path, encoding='utf-8')

        if '文本' not in df.columns:
            print(f"⚠️ 文件 {file_name} 缺少 '文本' 列，已跳过")
            continue

        texts = df['文本'].fillna("").astype(str).tolist()

        # 文本向量化（不使用任务列，只保留文本特征）
        X_text_vec = vectorizer.transform(texts)

        # 多输出模型预测
        preds = np.vstack([est.predict(X_text_vec) for est in estimators]).T  # shape: [N, 3]

        # 添加三列情绪值
        df['上证综合情绪值'] = preds[:, 0]
        df['沪深300情绪值'] = preds[:, 1]
        df['创业板情绪值'] = preds[:, 2]

        # 保存结果
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"❌ 文件 {file_name} 处理失败：{e}")

print("✅ 所有个股预测完成，结果已保存至：", output_folder)
