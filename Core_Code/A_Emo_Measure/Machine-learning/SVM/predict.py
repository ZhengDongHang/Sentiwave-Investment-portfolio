import os
import csv
import joblib
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message="Converting data to scipy sparse matrix")

# ---------------- 参数配置 ----------------
folder_path = '../../data/test_data/'   # 输入文件夹，包含多个个股 CSV 文件
save_model_path = 'svr_model.pkl'    # 训练好的 LinearSVR 模型路径
output_folder = 'result/'               # 输出目录
os.makedirs(output_folder, exist_ok=True)

# ---------------- 加载模型 ----------------
print("🔄 加载模型...")
vectorizer, model = joblib.load(save_model_path)

# ---------------- 获取所有 CSV 文件 ----------------
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
print(f"📂 共发现 {len(csv_files)} 个 CSV 文件")

# ---------------- 遍历每个文件并预测 ----------------
for file_name in tqdm(csv_files, desc="📊 处理中"):
    input_path = os.path.join(folder_path, file_name)
    output_path = os.path.join(output_folder, file_name)

    try:
        df = pd.read_csv(input_path, encoding='utf-8')

        if '文本' not in df.columns:
            print(f"⚠️ 文件 {file_name} 缺少 '文本' 列，已跳过")
            continue

        # 处理文本列
        texts = df['文本'].fillna("").astype(str).tolist()
        X_text_vec = vectorizer.transform(texts)

        # 模型预测（多标签回归，预测结果是二维数组）
        preds = model.predict(X_text_vec)

        # 分列写入预测值，假设3个目标维度
        df['上证综合情绪值'] = preds[:, 0]
        df['沪深300情绪值'] = preds[:, 1]
        df['创业板情绪值'] = preds[:, 2]

        # 保存结果
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"❌ 文件 {file_name} 处理失败：{e}")

print("✅ 所有个股预测完成，结果已保存至：", output_folder)
