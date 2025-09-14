import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 文件路径和标题
file_paths = [
    "ari_matrix/Test_for_the_instability_of_density.csv",
    "ari_matrix/Test_for_the_instability_of_feature.csv",
    "ari_matrix/Test_for_the_instability_of_representation.csv",
    "ari_matrix/Test_for_the_instability_of_shape.csv",
    "ari_matrix/Test_for_the_instability_of_Ours.csv"
]

titles = [
    "Instability of Density",
    "Instability of Feature",
    "Instability of Representation",
    "Instability of Shape",
    "Instability of Ours"
]

# 设置绘图风格
sns.set(style="white")
fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # 一行四列

for i, (file_path, title) in enumerate(zip(file_paths, titles)):
    # 读取 CSV
    df = pd.read_csv(file_path, index_col=0)

    # 绘制热力图
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=False,
                ax=axes[i], vmin=-1, vmax=1)

    axes[i].set_title(title, fontsize=12)
    axes[i].set_xticklabels(df.columns, rotation=45)
    axes[i].set_yticklabels(df.index, rotation=0)

# 调整布局
plt.tight_layout()
plt.savefig("ari_matrix/ari_matrix.png")
plt.show()
