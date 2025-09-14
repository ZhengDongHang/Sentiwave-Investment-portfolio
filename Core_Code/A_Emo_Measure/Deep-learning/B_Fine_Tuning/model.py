import torch
import torch.nn as nn
from transformers import BertModel


class BERTModel(nn.Module):
    def __init__(self, bert_model_name):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, local_files_only=True)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(p=0.1) # 随机“关闭”一部分神经元,提高泛化能力
        self.fc = nn.Linear(hidden_size, 3)
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]

        logits = self.fc(cls_output)  # [batch_size, 3]
        return logits

    def compute_loss(self, predictions, targets):
        return self.loss_fn(predictions, targets)

    def get_embedding(self, input_ids, attention_mask, token_type_ids):
        """
        提取 BERT 输出层前一层（用于嵌入式情绪分析）
        返回的是 encoder 输出的 [CLS] 向量或全局平均向量
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        # 使用 [CLS] token 的输出（即隐藏状态）
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_dim)
        return cls_embedding