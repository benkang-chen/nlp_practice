import json
import torch
from transformers import BertTokenizer
from model.bert_re import BertForRE

device = "cuda" if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    max_len = 512
    model_dir = './saved_model/model.pt'
    model = BertForRE.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    bert_path = '../bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    dev_data_path = './data/dev_data.json'
    with open(dev_data_path, encoding='utf8') as rf:
        for line in rf:
            data = json.loads(line)
            text = data["text"].lower()
            if len(text) <= max_len - 2:
                print(text)
                spo_list = data["spo_list"]
                for spo in spo_list:
                    print(f'{spo["subject"]} - {spo["predicate"]} - {spo["object"]}')
                chars = []
                for char in text:
                    chars.append(char)
                chars = ['[CLS]'] + chars + ['[SEP]']
                input_ids = [tokenizer.convert_tokens_to_ids(c) for c in chars]
                attention_mask = [1] * len(input_ids)
                input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)
                print(f'-----------------预测结果----------------')
                model.predict(text, chars, input_ids, attention_mask)
                print(f'=========================================')
