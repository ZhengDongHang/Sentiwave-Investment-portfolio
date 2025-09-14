import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForPreTraining
from torch.optim import AdamW
from tqdm import tqdm

from MLM import get_mlm_collator
from NSP import build_nsp_sentence_pairs
from Loss_Record import loss_record
from Data_Clean import clean_dataset

from transformers import logging
logging.set_verbosity_error() # 静默警告

DATA_DIR = "../../data/train_data"
SAVE_PATH = "../model/pre_train_model"
MODEL_NAME = "/data/public/fintechlab/zdh/Individual-Stock-Analysis/A_Emo_Measure/bert-base-chinese"
loss_log_path = "../loss/PreTrain_loss.csv"
BATCH_SIZE = 1024
EPOCHS = 3
MAX_LEN = 64

# 检查损失文件是否存在，如果存在则删除，防止重复写入
if os.path.exists(loss_log_path):
    os.remove(loss_log_path)
    print("原损失文件已删除")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = list(range(8)) # 使用8张3090显卡
os.makedirs(SAVE_PATH, exist_ok=True)


tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForPreTraining.from_pretrained(MODEL_NAME)
model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=1).to(DEVICE)

# 自定义 Dataset
class BertPretrainDataset(Dataset):
    def __init__(self, sentence_pairs, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sentence_pairs = sentence_pairs

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        text_a, text_b, label = self.sentence_pairs[idx]
        encoding = self.tokenizer.encode_plus(
                        text_a,
                        text_b,
                        max_length=self.max_len,
                        padding='max_length',
                        truncation=True,
                        return_overflowing_tokens=False,
                        return_tensors='pt'
                    )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'next_sentence_label': torch.tensor(label, dtype=torch.long)
        }

# 加载数据
def load_all_data(data_dir):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                try:
                    all_data.extend(json.load(f))
                except json.JSONDecodeError:
                    print(f"⚠️ 跳过无法解析的文件：{filename}")

    subset = all_data[:]  # *** 取一个子集（方便调试）*** #
    random.shuffle(subset)
    return subset

print("🟢 正在加载全量数据中...")
all_json_data = load_all_data(DATA_DIR) # 加载数据
print("✅ 加载全量数据成功")

print("🟢 正在进行数据清洗操作...")
all_json_data = clean_dataset(all_json_data) # 清洗数据
print("✅ 数据清洗成功")

print("🟢 正在进行NSP文本对的构建...")
sentence_pairs = build_nsp_sentence_pairs(all_json_data) # NSP数据构建
print("✅ NSP文本对的构建成功")

print("🟢 正在进行数据转化...")
dataset = BertPretrainDataset(sentence_pairs, tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print("✅ 数据转化成功")

collator = get_mlm_collator(tokenizer)
optimizer = AdamW(model.parameters(), lr=1e-5)

model.train()
print("🟢 开始训练...")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch['token_type_ids'].to(DEVICE)
        next_sentence_label = batch['next_sentence_label'].to(DEVICE)

        mlm_inputs = collator([
            {'input_ids': i, 'attention_mask': a}
            for i, a in zip(input_ids, attention_mask)
        ])
        mlm_inputs['token_type_ids'] = token_type_ids
        mlm_inputs['next_sentence_label'] = next_sentence_label
        mlm_inputs = {k: v.to(DEVICE) for k, v in mlm_inputs.items()}

        outputs = model(**mlm_inputs)
        loss = outputs.loss.mean()
        loss_record(loss_log_path, epoch, loop, loss) # 损失记录

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())

model.module.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"\n✅ 模型已保存到：{SAVE_PATH}")
