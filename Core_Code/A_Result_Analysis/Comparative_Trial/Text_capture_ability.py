import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# ==== 配置路径 ====
EMOTION_ROOT = '../data/Emotion_Data'
INDEX_CSV_PATH = '../data/Financial_Data/中国指数数据详细.csv'

# ==== 工具函数 ====
def merge_model_csvs(model_dir):
    """
    合并某个模型目录下的所有csv文件，返回按日期汇总的DataFrame
    保留列：日期，上证综合情绪值，沪深300情绪值，创业板情绪值
    """
    all_dfs = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                try:
                    df = pd.read_csv(path, encoding='utf-8')
                    all_dfs.append(df)
                except Exception as e:
                    print(f"读取文件{path}错误：{e}")
    if not all_dfs:
        print(f"{model_dir}无csv文件，返回空DataFrame")
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)
    expected_cols = ['日期', '上证综合情绪值', '沪深300情绪值', '创业板情绪值']
    for col in expected_cols:
        if col not in df_all.columns:
            print(f"警告：{model_dir}缺少列 {col}，补充空值")
            df_all[col] = np.nan
    df_all['日期'] = pd.to_datetime(df_all['日期'], errors='coerce')
    df_grouped = df_all.groupby('日期')[expected_cols[1:]].mean().reset_index()

    return df_grouped

def merge_integrated_csvs(model_dir):
    """
    专为 Integrated_task 模型处理：只读取日期和高维情绪变量
    """
    all_dfs = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                try:
                    df = pd.read_csv(path, encoding='utf-8')[['日期', '高维情绪变量']]
                    all_dfs.append(df)
                except Exception as e:
                    print(f"读取文件{path}错误：{e}")
    if not all_dfs:
        print(f"{model_dir}无有效csv文件，返回空DataFrame")
        return pd.DataFrame()
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all['日期'] = pd.to_datetime(df_all['日期'], errors='coerce')
    df_grouped = df_all.groupby('日期')['高维情绪变量'].first().reset_index()
    return df_grouped

def load_index_data(index_csv_path):
    df_index = pd.read_csv(index_csv_path, encoding='utf-8', dtype=str)
    df_index['交易日期'] = pd.to_datetime(df_index['交易日期'], errors='coerce')
    return df_index

def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # print('rmse:', rmse, 'mae:', mae, 'r2:', r2)
    return round(1/rmse, 3), round(1/mae, 3), round(r2, 3)

def main(emotion_root_dir, index_csv_path):
    df_index = load_index_data(index_csv_path)
    index_map = {
        '上证综合情绪值': '000001',
        '沪深300情绪值': '000300',
        '创业板情绪值': '399006'
    }
    results = {key: {} for key in index_map.keys()}

    category_dirs = [d for d in os.listdir(emotion_root_dir)
                     if os.path.isdir(os.path.join(emotion_root_dir, d))]

    for category in category_dirs:
        category_path = os.path.join(emotion_root_dir, category)
        model_dirs = [d for d in os.listdir(category_path)
                      if os.path.isdir(os.path.join(category_path, d))]

        for model in model_dirs:
            model_path = os.path.join(category_path, model)
            print(f"处理模型: {category}/{model}")
            col_name = model.strip()

            for emotion_col, index_code in index_map.items():
                df_index_sub = df_index[df_index['指数编号'] == index_code]

                # === 特殊处理 Integrated_task ===
                if col_name == 'Integrated_task':
                    df_model = merge_integrated_csvs(model_path)
                    if df_model.empty:
                        print(f"{col_name} 数据为空，跳过")
                        continue
                    df_merged = pd.merge(df_model, df_index_sub,
                                         left_on='日期', right_on='交易日期', how='inner')
                    if df_merged.empty:
                        print(f"{col_name} 与指数 {index_code} 无交集日期，跳过")
                        continue
                    try:
                        df_merged['高维情绪变量'] = df_merged['高维情绪变量'].apply(eval)
                        X = np.vstack(df_merged['高维情绪变量'].values)
                        y = df_merged['指数回报率'].astype(float).values
                        if X.shape[0] == 0 or len(y) == 0:
                            print(f"{col_name} 数据为空，跳过")
                            continue
                        model_lr = LinearRegression()
                        model_lr.fit(X, y)
                        r2 = r2_score(y, model_lr.predict(X))
                        if col_name not in results[emotion_col]:
                            results[emotion_col][col_name] = {}
                        results[emotion_col][col_name]['RMSE'] = np.nan
                        results[emotion_col][col_name]['MAE'] = np.nan
                        results[emotion_col][col_name]['R2'] = round(r2, 3)
                    except Exception as e:
                        print(f"{col_name} 高维变量处理失败：{e}")
                    continue

                # === 普通模型 ===
                df_model = merge_model_csvs(model_path)
                if df_model.empty:
                    print(f"{col_name} 数据为空，跳过")
                    continue
                df_merged = pd.merge(df_model, df_index_sub,
                                     left_on='日期', right_on='交易日期', how='inner')
                if df_merged.empty:
                    print(f"{col_name} 与指数 {index_code} 无交集日期，跳过")
                    continue

                if emotion_col not in df_merged.columns:
                    print(f"{col_name} 缺少列 {emotion_col}，跳过")
                    continue

                y_true = df_merged['指数回报率'].astype(float).values
                y_pred = df_merged[emotion_col].astype(float).values
                mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
                y_true_clean = y_true[mask]
                y_pred_clean = y_pred[mask]
                if len(y_true_clean) == 0:
                    print(f"{col_name} 无有效数据，跳过")
                    continue

                rmse_score, mae_score, r2 = calc_metrics(y_true_clean, y_pred_clean)
                if col_name not in results[emotion_col]:
                    results[emotion_col][col_name] = {}
                results[emotion_col][col_name]['RMSE'] = rmse_score
                results[emotion_col][col_name]['MAE'] = mae_score
                results[emotion_col][col_name]['R2'] = r2

    # ==== 保存结果 ====
    for emotion_col in results:
        df_res = pd.DataFrame(results[emotion_col]).T
        metrics = ['RMSE', 'MAE', 'R2']
        present_metrics = [m for m in metrics if m in df_res.columns]
        df_res = df_res[present_metrics].T

        baseline_model = None
        for col in df_res.columns:
            if col.strip() == 'Separated_task':
                baseline_model = col
                break

        if baseline_model:
            for col in df_res.columns:
                if col != baseline_model:
                    df_res[col] = df_res[col] - df_res[baseline_model]
            df_res[baseline_model] = 0
        else:
            print(f"警告：未找到基线模型 Separated_task，跳过差值处理")

        os.makedirs('情绪测度对比', exist_ok=True)
        save_path = f"情绪测度对比/{emotion_col.replace('情绪值','')}_指标结果.csv"
        print(f"保存结果文件: {save_path}")
        df_res.to_csv(save_path, encoding='utf-8-sig')

if __name__ == '__main__':
    main(EMOTION_ROOT, INDEX_CSV_PATH)