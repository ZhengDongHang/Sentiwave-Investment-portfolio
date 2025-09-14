import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from itertools import combinations
from tqdm import tqdm

# 设置路径
cluster_method = ('Test_for_the_instability_of_Ours')
folder = f"{cluster_method}/output"
files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])

# 读取所有文件为字典形式，按Stkcd索引
data = {}
for file in files:
    df = pd.read_csv(os.path.join(folder, file))
    df = df[['股票编号', '聚类标签']].dropna()
    df = df.set_index('股票编号').sort_index()
    data[file] = df

# 初始化ARI矩阵
ari_matrix = pd.DataFrame(index=files, columns=files)

# 计算ARI
for file1 in tqdm(files):
    for file2 in files:
        # 对齐索引，避免不同文件中Stkcd顺序或内容不同
        common_index = data[file1].index.intersection(data[file2].index)
        labels1 = data[file1].loc[common_index, '聚类标签']
        labels2 = data[file2].loc[common_index, '聚类标签']
        ari = adjusted_rand_score(labels1, labels2)
        ari_matrix.loc[file1, file2] = ari

# 保存结果
os.makedirs('ari_matrix', exist_ok=True)
ari_matrix.to_csv(os.path.join(f"ari_matrix/{cluster_method}.csv"), float_format='%.4f')
print("ARI矩阵已保存至 ari_matrix.csv")
