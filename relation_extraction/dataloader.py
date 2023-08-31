import json
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

# train_data_path = './data/train_data.json'
# dev_data_path = './data/dev_data.json'
# tokenizer_path = './bert-base-chinese'
# BATCH_SIZE = 32
# MAX_LEN = 512  # 输入模型的最大长度，不能超过config中n_ctx的值

start_tag, stop_tag = "<START>", "<STOP>"
tag2idx = {start_tag: 0, "O": 1, "B-SUB": 2, "I-SUB": 3, "B-OBJ": 4, "I-OBJ": 5, "B-BOTH": 6, "I-BOTH": 7, stop_tag: 8}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

relation2idx = {
    '祖籍': 0, '父亲': 1, '总部地点': 2, '出生地': 3, '目': 4, '面积': 5, '简称': 6,
    '上映时间': 7, '妻子': 8, '所属专辑': 9, '注册资本': 10, '首都': 11, '导演': 12,
    '字': 13, '身高': 14, '出品公司': 15, '修业年限': 16, '出生日期': 17, '制片人': 18,
    '母亲': 19, '编剧': 20, '国籍': 21, '海拔': 22, '连载网站': 23, '丈夫': 24,
    '朝代': 25, '民族': 26, '号': 27, '出版社': 28, '主持人': 29, '专业代码': 30,
    '歌手': 31, '作词': 32, '主角': 33, '董事长': 34, '成立日期': 35, '毕业院校': 36,
    '占地面积': 37, '官方语言': 38, '邮政编码': 39, '人口数量': 40, '所在城市': 41, '作者': 42,
    '作曲': 43, '气候': 44, '嘉宾': 45, '主演': 46, '改编自': 47, '创始人': 48
}
idx2relation = {idx: relation for relation, idx in relation2idx.items()}


def escape(text):
    return text.replace('+', '\\+').replace('*', '\\*').replace('?', '\\?').replace('{', '\\{').replace('}', '\\}'). \
        replace('(', '\\(').replace(')', '\\)').replace('[', '\\[').replace(']', '\\]')


def collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, tag_ids_list, attention_mask_list, real_lens_list, sub_mask_list, obj_mask_list, labels_list = [], [], [], [], [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        tag_ids_temp = instance["tag_ids"]
        attention_mask_temp = instance["attention_mask"]
        sub_mask_temp = instance["sub_mask"]
        obj_mask_temp = instance["obj_mask"]
        label_temp = instance["label"]
        real_len = instance["real_len"]
        # 添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        tag_ids_list.append(torch.tensor(tag_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
        sub_mask_list.append(torch.tensor(sub_mask_temp, dtype=torch.long))
        obj_mask_list.append(torch.tensor(obj_mask_temp, dtype=torch.long))
        labels_list.append(label_temp)
        real_lens_list.append(real_len)
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "tag_ids": pad_sequence(tag_ids_list, batch_first=True, padding_value=1),  # "O"对应的ID为1
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "sub_mask": pad_sequence(sub_mask_list, batch_first=True, padding_value=0),
            "obj_mask": pad_sequence(obj_mask_list, batch_first=True, padding_value=0),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "real_lens": real_lens_list}


class REDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_len):
        super(REDataset, self).__init__()
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.data_set = []
        with open(data_path, encoding='utf8') as rf:
            for line in rf:
                data = json.loads(line)
                text = data["text"].lower()
                real_len = len(text)
                if real_len <= max_len - 2:
                    try:
                        # 得到input_ids，
                        # s和o未必是分词工具分出来的词，因此要对query做标注才能抽取出正确的s、o，而考虑到分词可能切错边界，因此应该使用基于字的输入来标注
                        chars = []
                        for char in text:
                            chars.append(char)
                        input_ids = [tokenizer.convert_tokens_to_ids(c) for c in chars]
                        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
                        # 得到tag_ids
                        tags = ['O'] * real_len
                        spo_list = data["spo_list"]
                        subjects = set()
                        objects = set()
                        for spo in spo_list:
                            obj = spo["object"]
                            objects.add(obj)
                            sub = spo["subject"]
                            subjects.add(sub)
                        # 去掉object
                        subjects_new = subjects - objects
                        for sub in subjects_new:
                            matches = re.finditer(escape(sub), text)
                            for match in matches:
                                start, end = match.start(), match.end()
                                tags[start] = 'B-SUB'
                                tags[start + 1:end] = ['I-SUB'] * (end - start - 1)
                        objects_new = objects - subjects
                        for obj in objects_new:
                            matches = re.finditer(escape(obj), text)
                            for match in matches:
                                start, end = match.start(), match.end()
                                tags[start] = 'B-OBJ'
                                tags[start + 1:end] = ['I-OBJ'] * (end - start - 1)
                        both = objects & subjects
                        for b in both:
                            matches = re.finditer(escape(b), text)
                            for match in matches:
                                start, end = match.start(), match.end()
                                tags[start] = 'B-BOTH'
                                tags[start + 1:end] = ['I-BOTH'] * (end - start - 1)
                        tags = ['O'] + tags + ['O']
                        tag_ids = [tag2idx[tag] for tag in tags]
                        # 得到subject mask和object mask
                        for spo in spo_list:
                            sub_mask = [0] * real_len
                            obj_mask = [0] * real_len
                            predicate = spo["predicate"]
                            obj = spo["object"].lower()
                            sub = spo["subject"].lower()
                            start_sub, end_sub = re.search(escape(sub), text).span()
                            sub_mask[start_sub:end_sub] = [1] * (end_sub - start_sub)
                            sub_mask = [0] + sub_mask + [0]
                            start_obj, end_obj = re.search(escape(obj), text).span()
                            obj_mask[start_obj:end_obj] = [1] * (end_obj - start_obj)
                            obj_mask = [0] + obj_mask + [0]
                            label = relation2idx[predicate]
                            assert len(input_ids) == len(tag_ids) == len(sub_mask) == len(obj_mask) == real_len + 2
                            self.data_set.append(
                                {"input_ids": input_ids, "tag_ids": tag_ids, "attention_mask": [1] * len(input_ids),
                                 "sub_mask": sub_mask, "obj_mask": obj_mask, "label": label, "real_len": real_len + 2})
                    except Exception as e:
                        print(e)
                        print(data)
                        print(text)
                        print(spo)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


def get_dataloader(data_path, tokenizer_path='../bert-base-chinese', max_len=512, batch_size=6, shuffle=True):
    dataset = REDataset(data_path, tokenizer_path, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


if __name__ == '__main__':
    train_data_path = './data/dev_data.json'
    data_loader = get_dataloader(train_data_path, shuffle=False)
    data_iter = iter(data_loader)
    print(next(data_iter))
