import papermill as pm
from concurrent.futures import ThreadPoolExecutor
import os

# 映射表
emo_index_map = ['上证综合情绪值', '沪深300情绪值', '创业板情绪值']
stage_map = ['计算矩阵', '读取矩阵']
model_map = [
    'Emo-Dict/DLUT', 'Emo-Dict/Bian', 'Emo-Dict/Jiang',
    'Machine-learning/LR', 'Machine-learning/RF', 'Machine-learning/SVM',
    'Deep-learning/Separated_task'
]

def run_notebook(emo_index_id, stage_id, model_id, data_number):
    emo_name = emo_index_map[emo_index_id]
    stage_name = stage_map[stage_id]
    model_name = model_map[model_id]

    # 构造输出路径，自动创建文件夹
    os.makedirs(f"output/{data_number}/{model_name}", exist_ok=True)
    output_path = f"output/{data_number}/{model_name}/{emo_name}.ipynb"

    print(f"Start running: emo_index_id={emo_index_id} ({emo_name}), "
          f"stage_id={stage_id} ({stage_name}), model_id={model_id} ({model_name})")

    pm.execute_notebook(
        "Clustering_Based_on_Separation.ipynb",
        output_path,
        parameters={
            "emo_index_id": emo_index_id,
            "stage_id": stage_id,
            "model_id": model_id,
            "data_number": data_number
        }
    )

    print(f"Finished running: {data_number}-{model_name}-{emo_name}")

def main():
    stage_id = 0
    model_ids = [0, 1, 2, 3, 4, 5, 6]
    emo_index_ids = [0]
    data_numbers = [20, 50, 100, 200]
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for data_number in data_numbers:
            for model_id in model_ids:
                for emo_id in emo_index_ids:
                    futures.append(executor.submit(run_notebook, emo_id, stage_id, model_id, data_number))

        # 等待所有任务完成
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
