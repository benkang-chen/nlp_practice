import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertPreTrainedModel


class SentenceBert(BertPreTrainedModel):
    def __init__(self, config, method='mean_pooling', use_cosine_embedding_loss=False):
        super(SentenceBert, self).__init__(config)
        self.bert = BertModel(config)
        self.fc = nn.Linear(3 * 768, 2)
        self.method = method
        self.use_cosine_embedding_loss = use_cosine_embedding_loss

    def get_sentence_embedding(self, context_1, mask_1, context_2, mask_2):
        outputs_1 = self.bert(context_1, attention_mask=mask_1, output_hidden_states=True)
        last_hidden_state_1 = outputs_1[0]  # [batch_size, seq_len, 768]
        # 序列的第一个token (cls) 的最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的，
        # 这个输出不是对输入的语义内容的一个很好的总结，对于整个输入序列的隐藏状态序列的平均化或池化可以更好的表示一句话。
        pooler_1 = outputs_1[1]  # [batch_size, 768]

        outputs_2 = self.bert(context_2, attention_mask=mask_2, output_hidden_states=True)
        last_hidden_state_2 = outputs_2[0]  # [batch_size, seq_len, 768]
        pooler_2 = outputs_2[1]  # [batch_size, 768]

        if self.method == 'pooler':
            return pooler_1, pooler_2
        if self.method == 'max_pooling':
            out_1 = last_hidden_state_1.permute(0, 2, 1)
            out_1 = F.max_pool1d(out_1, out_1.size(2)).squeeze(2)
            out_2 = last_hidden_state_2.permute(0, 2, 1)
            out_2 = F.max_pool1d(out_2, out_2.size(2)).squeeze(2)
            return out_1, out_2
        if self.method == 'mean_pooling':
            out_1 = last_hidden_state_1.permute(0, 2, 1)
            out_1 = nn.AvgPool1d(out_1.size(2))(out_1).squeeze(2)
            out_2 = last_hidden_state_2.permute(0, 2, 1)
            out_2 = nn.AvgPool1d(out_2.size(2))(out_2).squeeze(2)
            return out_1, out_2

    def forward(self, context_1, mask_1, context_2, mask_2):
        embed_1, embed_2 = self.get_sentence_embedding(context_1, mask_1, context_2, mask_2)
        if self.use_cosine_embedding_loss:
            return embed_1, embed_2
        vec = torch.cat((embed_1, embed_2, torch.abs(embed_1 - embed_2)), 1)  # [batch_size, 3*768]
        logit = self.fc(vec)
        return logit

    def predict(self, context_1, mask_1, context_2, mask_2, threshold):
        embed_1, embed_2 = self.get_sentence_embedding(context_1, mask_1, context_2, mask_2)
        simi = torch.cosine_similarity(embed_1, embed_2)
        pred = torch.where(simi >= threshold, 1, 0)
        pred = pred.cpu().numpy()
        return pred
