import papermill as pm
import os

# 创建输出文件夹（如果不存在）
os.makedirs("output", exist_ok=True)

# 定义扰动级别，单位为 BP（1BP = 0.0001）
bp_levels = [0, 1, 10, 50]
# 遍历每个扰动等级
for bp in bp_levels:
    output_path = f"output/Clustering_Based_on_Embedding_{bp}BP.ipynb"

    print(f"Running notebook for {bp}BP perturbation...")

    # 执行notebook，传入参数 bp_level
    pm.execute_notebook(
        input_path="Clustering_Based_on_Embedding.ipynb",
        output_path=output_path,
        parameters={"bp_level": bp}
    )

    print(f"Saved to {output_path}")
