import pandas as pd
import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class SentenceBertDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_len):
        super(SentenceBertDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len
        self.data_set = []

        data = pd.read_csv(data_path, low_memory=False)
        for _, row in data.iterrows():
            sentence_1 = row['query1']
            sentence_2 = row['query2']
            label = row['label']

            tokens_1 = self.tokenizer.tokenize(sentence_1)
            if len(tokens_1) > self.max_len - 2:
                tokens_1 = tokens_1[:self.max_len]
            input_ids_1 = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens_1 + ['[SEP]'])
            mask_1 = [1] * len(input_ids_1)

            tokens_2 = self.tokenizer.tokenize(sentence_2)
            if len(tokens_2) > self.max_len - 2:
                tokens_2 = tokens_2[:self.max_len]
            input_ids_2 = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens_2 + ['[SEP]'])
            mask_2 = [1] * len(input_ids_2)

            self.data_set.append({
                "input_ids_1": input_ids_1,
                "mask_1": mask_1,
                "input_ids_2": input_ids_2,
                "mask_2": mask_2,
                "label": label
            })

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


def sentence_bert_collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list_1, attention_mask_list_1, input_ids_list_2, attention_mask_list_2, labels_list = [], [], [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp_1 = instance["input_ids_1"]
        attention_mask_temp_1 = instance["mask_1"]
        input_ids_temp_2 = instance["input_ids_2"]
        attention_mask_temp_2 = instance["mask_2"]
        label_temp = instance["label"]
        # 将input_ids_temp和attention_mask_temp添加到对应的list中
        input_ids_list_1.append(torch.tensor(input_ids_temp_1, dtype=torch.long))
        attention_mask_list_1.append(torch.tensor(attention_mask_temp_1, dtype=torch.long))
        input_ids_list_2.append(torch.tensor(input_ids_temp_2, dtype=torch.long))
        attention_mask_list_2.append(torch.tensor(attention_mask_temp_2, dtype=torch.long))
        labels_list.append(label_temp)
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids_1": pad_sequence(input_ids_list_1, batch_first=True, padding_value=0),
            "attention_mask_1": pad_sequence(attention_mask_list_1, batch_first=True, padding_value=0),
            "input_ids_2": pad_sequence(input_ids_list_2, batch_first=True, padding_value=0),
            "attention_mask_2": pad_sequence(attention_mask_list_2, batch_first=True, padding_value=0),
            "labels": torch.tensor(labels_list, dtype=torch.long)}


class SimCSEDatasetForTrain(Dataset):
    def __init__(self, data_path, tokenizer_path, max_len):
        super(SimCSEDatasetForTrain, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len

        self.data_set = []
        data = pd.read_csv(data_path, low_memory=False)
        for _, row in data.iterrows():
            sentence = row['sentence']
            tokens = self.tokenizer.tokenize(sentence)
            if len(tokens) > self.max_len - 2:
                tokens = tokens[:self.max_len]
            input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
            mask = [1] * len(input_ids)
            for k in range(2):
                self.data_set.append({"input_ids": input_ids, "mask": mask})

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


def sim_cse_collate_fn_train(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, attention_mask_list = [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        attention_mask_temp = instance["mask"]
        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0)}


def get_dataloader(data_path, tokenizer_path, max_len, batch_size=6, shuffle=True, dataset_type='sentence_bert'):
    if dataset_type == 'sim_cse_train':
        dataset = SimCSEDatasetForTrain(data_path, tokenizer_path, max_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=sim_cse_collate_fn_train)
    else:
        dataset = SentenceBertDataset(data_path, tokenizer_path, max_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=sentence_bert_collate_fn)


if __name__ == '__main__':
    data_loader = get_dataloader(data_path='./data/train_update_1.csv',
                                 tokenizer_path='../bert-base-chinese',
                                 max_len=256,
                                 dataset_type='sim_cse_train')
    n = data_loader.size if hasattr(data_loader, 'size') else len(data_loader)
    print(n)
    data_iter = iter(data_loader)
    print(next(data_iter))
