import os
import jieba
import logging
import pandas as pd
from tqdm import tqdm
from loguru import logger
from collections import Counter


class Vocab:
    def __init__(self, fields: list, data_path=None, vocab_file='./data/vocab.txt', min_word_frequency=5):
        jieba.setLogLevel(logging.INFO)
        self.BOS_TAG = '<bos>'  # 序列开始字符
        self.EOS_TAG = '<eos>'  # 句子结束字符
        self.UNK_TAG = '<unk>'  # 特殊字符
        self.PAD_TAG = '<pad>'  # 填充字符
        self.BOS_IDX = 0
        self.EOS_IDX = 1
        self.UNK_IDX = 2
        self.PAD_IDX = 3
        if data_path is None and vocab_file is None:
            logger.error("data dir or vocab file path is not set.")
        if not os.path.isfile(vocab_file):
            self.build_vocab(data_path, fields, min_word_frequency)
        with open(file=vocab_file, encoding='utf-8') as f:
            self.word2id = {word.strip('\n'): idx for idx, word in enumerate(f.readlines())}
            self.id2word = {idx: word for word, idx in self.word2id.items()}

    @staticmethod
    def tokenizer(text):
        token = [tok for tok in jieba.cut(text)]
        return token

    def build_vocab(self, data_path, fields, min_word_frequency):
        def generate_lines(word_list):
            for word in word_list:
                yield word + '\n'
        data = pd.read_csv(data_path, sep='\t')
        words = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc='build vocab'):
            for field in fields:
                sentence = row[field]
                token = self.tokenizer(sentence)
                words.extend(token)
        counter = Counter(words).most_common()
        words = [self.BOS_TAG, self.EOS_TAG, self.UNK_TAG, self.PAD_TAG] + \
                [word for word, count in counter if count > min_word_frequency]
        with open('data/vocab.txt', 'w', encoding='utf-8') as f:
            f.writelines(generate_lines(words))


if __name__ == '__main__':
    vocab = Vocab(['trg', 'src'], data_path='data/train.tsv', min_word_frequency=10)
