import jieba
import os
import csv
import chardet
import pandas as pd
from tqdm import tqdm

# ==========================================
# 配置参数
# ==========================================
Emo_Dict = ['DLUT', 'Bian', 'Jiang'][0]  # 选择词典：DLUT / Bian / Jiang
input_folder = '../data/test_data/'   # 输入文件夹：每只个股一个 CSV 文件
output_folder = f'results/{Emo_Dict}/'  # 输出目录
os.makedirs(output_folder, exist_ok=True)

# ==========================================
# 加载情绪词典
# ==========================================
def load_words(dict_folder):
    pos_path = os.path.join(dict_folder, 'emo_dict', 'positive_words.txt')
    neg_path = os.path.join(dict_folder, 'emo_dict', 'negative_words.txt')

    def detect_encoding(file_path):
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            return result['encoding']

    pos_encoding = detect_encoding(pos_path)
    neg_encoding = detect_encoding(neg_path)

    with open(pos_path, 'r', encoding=pos_encoding, errors='ignore') as f:
        positive_words = set(f.read().splitlines())
    with open(neg_path, 'r', encoding=neg_encoding, errors='ignore') as f:
        negative_words = set(f.read().splitlines())

    return positive_words, negative_words

positive_words, negative_words = load_words(Emo_Dict)

# ==========================================
# 情绪得分计算函数
# ==========================================
def calculate_sentiment(text):
    if isinstance(text, float):
        text = str(text)
    words = jieba.lcut(text)
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    total_count = len(words)
    if total_count == 0 or (pos_count + neg_count) == 0:
        return 0
    pos_ratio = pos_count / total_count
    neg_ratio = neg_count / total_count
    return (pos_ratio - neg_ratio) / (pos_ratio + neg_ratio)

# ==========================================
# 遍历每只个股 CSV 文件并处理
# ==========================================
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
print(f"📂 共发现 {len(csv_files)} 个 CSV 文件")

for file_name in tqdm(csv_files, desc="📊 正在处理"):
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    try:
        df = pd.read_csv(input_path, encoding='utf-8')

        if '文本' not in df.columns:
            print(f"⚠️ 文件 {file_name} 缺少 '文本' 列，已跳过")
            continue

        texts = df['文本'].fillna("").astype(str).tolist()
        sentiments = [calculate_sentiment(text) for text in texts]

        # 复制三列
        df['上证综合情绪值'] = sentiments
        df['沪深300情绪值'] = sentiments
        df['创业板情绪值'] = sentiments

        df.to_csv(output_path, index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"❌ 文件 {file_name} 处理失败：{e}")

print(f"✅ 情绪词典分析（{Emo_Dict}）完成，结果已保存至 {output_folder}")
