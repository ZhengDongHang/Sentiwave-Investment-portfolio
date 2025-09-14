import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from model import BERTModel
from dataset import Dataset_ForPred, collate_fn_ForPred

import multiprocessing as mp

# 确定化预测结果
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ========= 配置 ==========
Choice = ['Separated', 'Integrated'][1]
DATA_DIR = "../../data/test_data"
MODEL_NAME = "../model/pre_train_model"
MODEL_PATH = f"../model/Fine-tuning-BERT/model.pt"
SAVE_DIR = f"../results/{Choice}_task"
BATCH_SIZE = 512
MAX_LEN = 64
DEVICE_IDS = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 个 GPU
NUM_PROCESSES = len(DEVICE_IDS)

os.makedirs(SAVE_DIR, exist_ok=True)

# ========= 每个进程的工作函数 ==========
def process_file(args):
    device_id, file_name = args
    torch.cuda.set_device(device_id)
    print(f"🚀 GPU {device_id} 正在处理文件：{file_name}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = BERTModel(bert_model_name=MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=f"cuda:{device_id}"))
    model = model.to(f"cuda:{device_id}")
    model.eval()

    input_path = os.path.join(DATA_DIR, file_name)
    output_path = os.path.join(SAVE_DIR, file_name)

    try:
        df = pd.read_csv(input_path, encoding="utf-8")
    except:
        df = pd.read_csv(input_path, encoding="gbk")

    if "文本" not in df.columns:
        print(f"⚠️ 文件 {file_name} 缺少“文本”列，跳过。")
        return

    raw_data = df[["文本"]].rename(columns={"文本": "text"}).to_dict(orient="records")
    dataset = Dataset_ForPred(raw_data, tokenizer, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda x: collate_fn_ForPred(x, tokenizer, MAX_LEN))

    records = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, texts in dataloader:
            input_ids = input_ids.to(f"cuda:{device_id}")
            attention_mask = attention_mask.to(f"cuda:{device_id}")
            token_type_ids = token_type_ids.to(f"cuda:{device_id}")

            if Choice == 'Separated':
                logits = model({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                })
                preds = logits.cpu()
                for i in range(len(texts)):
                    records.append({
                        "文本": texts[i].replace(" ", ""),
                        "上证综合情绪值": round(preds[i][0].item(), 3),
                        "沪深300情绪值": round(preds[i][1].item(), 3),
                        "创业板情绪值": round(preds[i][2].item(), 3),
                    })
            else:
                hidden_states = model.get_embedding(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                hidden_states = hidden_states.cpu()
                for i in range(len(texts)):
                    records.append({
                        "文本": texts[i].replace(" ", ""),
                        "高维情绪变量": hidden_states[i].tolist()
                    })

    df_embed = pd.DataFrame(records)
    df_result = pd.concat([df.reset_index(drop=True), df_embed], axis=1)
    df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ GPU {device_id} 完成文件处理：{file_name}")

# ========= 主控制函数 ==========
if __name__ == "__main__":
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    print(f"🔎 共检测到 {len(csv_files)} 个待处理文件")

    task_queue = []
    for idx, file_name in enumerate(csv_files):
        device_id = DEVICE_IDS[idx % len(DEVICE_IDS)]  # 循环分配 GPU
        task_queue.append((device_id, file_name))

    with mp.Pool(processes=len(DEVICE_IDS)) as pool:
        list(tqdm(pool.imap_unordered(process_file, task_queue), total=len(task_queue)))