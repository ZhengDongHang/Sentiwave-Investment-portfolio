import re
import random

def split_text(text):
    # 使用中文标点符号进行分句
    sentences = re.split(r'[。！？.!,?]', text)
    # 去除短句和空白
    return [s.strip() for s in sentences if len(s.strip()) >= 8]

def build_nsp_sentence_pairs(data_list):
    pairs = []
    for item in data_list:
        texts = item.get("guba_data", [])
        for text in texts:
            sentences = split_text(text)
            if len(sentences) < 2:
                continue
            for i in range(len(sentences) - 1):
                # 正样本
                pairs.append((sentences[i], sentences[i + 1], 1))
                # 负样本（随机一句）
                neg_idx = random.randint(0, len(sentences) - 1)
                if neg_idx != i:
                    pairs.append((sentences[i], sentences[neg_idx], 0))
    return pairs
