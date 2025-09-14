import os
import time
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from model import BERTModel
from dataset import Dataset, collate_fn
from utils import load_all_data, loss_record


# 配置
DATA_DIR = "../../data/train_data"
MODEL_NAME = "../model/pre_train_model"
SAVE_PATH = f"../model/Fine-tuning-BERT"
loss_log_path = f"../loss/loss.csv"
BATCH_SIZE = 4096
EPOCHS = 3
MAX_LEN = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = list(range(8))
os.makedirs(SAVE_PATH, exist_ok=True)

if os.path.exists(loss_log_path):
    os.remove(loss_log_path)

print("🟢 加载 tokenizer 和模型...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

# 调用BERT学习模型
model = BERTModel(bert_model_name=MODEL_NAME)

model = DataParallel(model, device_ids=device_ids).to(DEVICE)
print("✅ 模型加载完成")

print("🟢 加载数据...")
data = load_all_data(DATA_DIR)
dataset = Dataset(data, tokenizer, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer, max_len=MAX_LEN))
print("✅ 数据加载完成，总样本数：", len(dataset))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

print("🟢 开始训练...")
model.train()
start_time = time.time() # 记录开始时间
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
    for step, (input_ids, attention_mask, token_type_ids, labels, task_types) in loop:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # 构建一个 dict 传入模型
        logits = model({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "task_types": task_types  # 注意：仍是 list
        })

        loss = model.module.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())
        elapsed_time = time.time() - start_time # 实现计数功能
        loss_record(loss_log_path, epoch, step, loss.item(), elapsed_time)

# 保存模型
torch.save(model.module.state_dict(), os.path.join(SAVE_PATH, "model.pt"))
tokenizer.save_pretrained(SAVE_PATH)
print(f"\n✅ 模型已保存至：{SAVE_PATH}")
