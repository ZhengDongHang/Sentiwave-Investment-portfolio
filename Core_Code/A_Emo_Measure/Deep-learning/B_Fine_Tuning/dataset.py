import torch
from torch.utils.data import Dataset


class Dataset(Dataset): # 训练用数据集，无需date
    def __init__(self, data, tokenizer, max_len):
        self.samples = []
        for record in data[:]:
            index = torch.tensor(record["index"], dtype=torch.float)
            for text in record.get("guba_data", []):
                self.samples.append((text, index, "guba"))

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, tokenizer, max_len):
    texts, labels, task_types = zip(*batch)
    encoding = tokenizer(
        list(texts),
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    labels = torch.stack(labels)
    return encoding["input_ids"], encoding["attention_mask"], encoding["token_type_ids"], labels, list(task_types)


# ✅ 用于预测的 Dataset，不使用日期，只用文本
class Dataset_ForPred(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.samples = [record["text"] for record in data]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn_ForPred(batch, tokenizer, max_len):
    encoding = tokenizer(
        batch,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoding["input_ids"], encoding["attention_mask"], encoding["token_type_ids"], batch  # batch 是原始文本