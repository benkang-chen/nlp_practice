from text_summary.vocab import Vocab


class Word2Sequence:

    def __init__(self, vocab_file='./data/vocab.txt'):
        self.vocab = Vocab(['trg', 'src'], vocab_file=vocab_file)

    def transform(self, words, max_len=None, add_bos=True, add_eos=False):
        """
        把词转成序列
        :param add_bos:
        :param words:
        :param max_len:
        :param add_eos:
        :return:
        """
        # if add_bos:
        #     max_len -= 1
        # if add_eos:
        #     max_len -= 1  # 为了输入和目标句子的长度都等于设置的max_len
        if max_len is not None:
            if len(words) > max_len:
                words = words[:max_len]
            else:
                words += [self.vocab.PAD_TAG] * (max_len - len(words))
        if add_bos:
            words.insert(0, self.vocab.BOS_TAG)
        if add_eos:
            if words[-1] == self.vocab.PAD_TAG:
                words.insert(words.index(self.vocab.PAD_TAG), self.vocab.EOS_TAG)  # 在PAD之前添加EOS
            else:
                words += [self.vocab.EOS_TAG]  # 数字字符串中没有PAD，在最后添加EOS
        return [self.vocab.word2id.get(word, self.vocab.UNK_IDX) for word in words]

    def inverse_transform(self, sequence):  # 把数字序列转化为数字字符串
        results = []
        for index in sequence:
            result = self.vocab.id2word.get(index, self.vocab.UNK_TAG)
            if result != self.vocab.EOS_TAG:
                results.append(result)
            else:
                break
        return results

    def __len__(self):
        return len(self.vocab.word2id)


if __name__ == '__main__':
    word_sequence = Word2Sequence()
    str1 = ['你好', '中国']
    words = word_sequence.transform(str1, max_len=4, add_eos=True)
    print(words)
