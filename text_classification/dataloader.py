import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer

multi_label_binarizer = MultiLabelBinarizer()
all_categories = []
for infile in ["./data/multi-classification-train.txt", "./data/multi-classification-test.txt"]:
    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            cate = line.strip().split(" ", maxsplit=1)[0].split("|")
            all_categories.append(cate)
multi_label_binarizer.fit(all_categories)


class MultilabelDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_len):
        super(MultilabelDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len
        self.data_set = []
        with open(data_path, 'r', encoding='utf8') as rf:
            for line in rf:
                cate, sent = line.strip().split(" ", maxsplit=1)
                cate = cate.split("|")
                label = multi_label_binarizer.transform([cate])[0].tolist()
                tokens = self.tokenizer.tokenize(sent)
                if len(tokens) > self.max_len - 2:
                    tokens = tokens[:self.max_len]
                input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
                mask = [1] * len(input_ids)
                self.data_set.append({"input_ids": input_ids, "mask": mask, "label": label})

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


def multi_label_collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, attention_mask_list, labels_list = [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        attention_mask_temp = instance["mask"]
        label_temp = instance["label"]
        # 添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
        labels_list.append(label_temp)
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "labels": torch.tensor(labels_list, dtype=torch.long)}


def get_dataloader(data_path, tokenizer_path, max_len, batch_size=6, task_type='multi_label', shuffle=True):
    if task_type == 'multi_label':
        dataset = MultilabelDataset(data_path, tokenizer_path, max_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=multi_label_collate_fn)


if __name__ == '__main__':
    data_loader = get_dataloader(data_path='./data/multi-classification-train.txt',
                                 tokenizer_path='../bert-base-chinese',
                                 max_len=256)
    n = data_loader.size if hasattr(data_loader, 'size') else len(data_loader)
    print(n)
    data_iter = iter(data_loader)
    print(next(data_iter))
