import papermill as pm
import os

# 建立 experiment_id -> experiment_name 的映射字典
experiment_map = {
    0: 'Ours',
    1: 'BERT + DTW',
    2: 'BERT + Density',
    3: 'BERT + Features',
    4: 'BERT + Representations'
}

# 创建输出文件夹（如果不存在）
os.makedirs("output", exist_ok=True)

# 遍历 experiment_id
# Methods = [0, 1, 2, 3, 4]
Methods = [4]
for experiment_id in Methods:
    experiment_name = experiment_map[experiment_id]  # 通过映射得到名字
    output_path = f"output/{experiment_name}.ipynb"  # 用名字生成文件名

    print(f"Running notebook of {experiment_name}")

    # 执行 notebook，传入 experiment_id
    pm.execute_notebook(
        input_path="Investment_portfolio.ipynb",
        output_path=output_path,
        parameters={"experiment_id": experiment_id}
    )

    print(f"Saved to {output_path}")
