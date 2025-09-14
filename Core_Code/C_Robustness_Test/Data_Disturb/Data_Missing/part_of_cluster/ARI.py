import pandas as pd
from sklearn.metrics import adjusted_rand_score
import numpy as np
import os

# 文件名及标签
file_names = ['output/20.csv', 'output/50.csv', 'output/100.csv', 'output/183.csv']
labels = ['20', '50', '100', '183']

# 检查文件是否存在
for f in file_names:
    if not os.path.exists(f):
        raise FileNotFoundError(f"文件不存在: {f}")

# 读取文件
dfs = [pd.read_csv(f) for f in file_names]

# 初始化 ARI 相似度矩阵
n = len(dfs)
ari_matrix = np.zeros((n, n))

# 计算 ARI 相似度
for i in range(n):
    for j in range(n):
        # 找出公司编号交集，并对齐聚类标签
        merged = pd.merge(dfs[i], dfs[j], on='公司编号', suffixes=('_i', '_j'))
        if len(merged) == 0:
            score = np.nan
        else:
            score = adjusted_rand_score(merged['聚类标签_i'], merged['聚类标签_j'])
        ari_matrix[i, j] = score

# 转为 DataFrame 并保存为 CSV
ari_df = pd.DataFrame(ari_matrix, index=labels, columns=labels)
ari_df.to_csv('相似度检验结果.csv', encoding='utf-8-sig', index=True)

print("✅ 相似度检验结果已保存为：相似度检验结果.csv")
