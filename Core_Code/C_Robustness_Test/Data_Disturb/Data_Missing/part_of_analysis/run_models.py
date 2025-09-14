import papermill as pm
from concurrent.futures import ThreadPoolExecutor
import os

# 映射表
emo_index_map = ['上证综合情绪值', '沪深300情绪值', '创业板情绪值']
model_map = [
    'Emo-Dict/DLUT', 'Emo-Dict/Bian', 'Emo-Dict/Jiang',
    'Machine-learning/LR', 'Machine-learning/RF', 'Machine-learning/SVM',
    'Deep-learning/BERT', 'Deep-learning/Ours'
]

def run_notebook(model_id, emo_id, N):
    emo_name = emo_index_map[emo_id]
    model_name = model_map[model_id]

    # 构造输出路径，自动创建文件夹
    os.makedirs(f"output/Models/{N}/{model_name}", exist_ok=True)
    output_path = f"output/Models/{N}/{model_name}/{emo_name}.ipynb"

    print(f"Start running: model_name = {model_name} -- emo_name = {emo_name}")

    pm.execute_notebook(
        "Model_Comparison.ipynb",
        output_path,
        parameters={
            "EMO_INDEX": emo_id,
            "MODEL_INDEX": model_id,
            "N": N
        }
    )

    print(f"Finished running: {model_name}-{emo_name}")

def main():
    N_list = {20, 50, 100, 200}
    model_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    emo_index_ids = [0, 1, 2]
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for N in N_list:
            for model_id in model_ids:
                for emo_id in emo_index_ids:
                    futures.append(executor.submit(run_notebook, model_id, emo_id, N))

        # 等待所有任务完成
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
