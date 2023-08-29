import torch
import pandas as pd
from tqdm import tqdm
from vocab import Vocab
from uitls import Word2Sequence
from torch.utils.data import Dataset, DataLoader

word2seq = Word2Sequence()


class Seq2SeqDataset(Dataset):

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, sep='\t', low_memory=False)

    def __getitem__(self, idx):
        target = self.data.at[idx, 'trg']
        target_token = Vocab.tokenizer(target)
        source = self.data.at[idx, 'src']
        source_token = Vocab.tokenizer(source)

        return target_token, source_token, len(target_token), len(source_token)

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    targets, sources, target_lengths, source_lengths = zip(*batch)
    targets = torch.LongTensor(
        [word2seq.transform(target, max_len=max(target_lengths), add_eos=True) for target in targets]
    )

    sources = torch.LongTensor(
        [word2seq.transform(source, max_len=max(source_lengths), add_eos=True) for source in sources]
    )
    target_lengths = torch.LongTensor(target_lengths)
    source_lengths = torch.LongTensor(source_lengths)
    return sources, targets, source_lengths, target_lengths


def get_dataloader(data_path='./data/train.tsv', batch_size=6):
    dataset = Seq2SeqDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


if __name__ == '__main__':
    data_loader = get_dataloader()
    data_iter = iter(data_loader)
    pbar = tqdm(data_iter)
    for s, t, s_l, t_l in pbar:
        print(s)
        break

