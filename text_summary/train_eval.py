import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from text_summary.dataloader import get_dataloader
from seq2seq.model.seq2seq_lstm import Seq2Seq
from seq2seq.model.seq2seq_gru_att import Seq2Seq_Att
from transformer.model.transformer import Transformer
from text_summary.vocab import Vocab
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    model_type = 'transformer'
    vocab = Vocab(['trg', 'src'], data_path='data/train.tsv')
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    input_dim = len(vocab.id2word)
    output_dim = len(vocab.id2word)
    num_layers = 2
    n_epochs = 10
    clip = 1
    if model_type == 'seq2seq':
        tensorboard_dir = 'Seq2seq/runs/seq2seq'
        model = Seq2Seq(input_dim, output_dim, device, num_layers=num_layers).to(device)
    elif model_type == 'seq2seq_att':
        tensorboard_dir = 'Seq2seq/runs/seq2seq_att'
        model = Seq2Seq_Att(input_dim, output_dim, device, num_layers=1).to(device)
    else:
        tensorboard_dir = 'transformer/runs/transformer'
        hidden_dim = 512
        encoder_layers = 6
        decoder_layers = 6
        encoder_heads = 8
        decoder_heads = 8
        encoder_pf_dim = 2048
        decoder_pf_dim = 2048
        encoder_dropout = 0.1
        decoder_dropout = 0.1
        model = Transformer(input_dim, hidden_dim, encoder_layers, encoder_heads, encoder_pf_dim, encoder_dropout,
                            output_dim, decoder_layers, decoder_heads, decoder_pf_dim, decoder_dropout,
                            vocab.PAD_IDX,
                            device).to(device)



    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)


    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    # we ignore the loss whenever the target token is a padding token.
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)

    loss_vals = []
    loss_vals_eval = []
    train_iter = iter(get_dataloader(data_path='data/train.tsv'))
    source, target, _, _ = next(train_iter)
    writer = SummaryWriter(tensorboard_dir)
    # model.add_graph = True
    # writer.add_graph(model.eval(), input_to_model=[source.to(device), target.to(device)])  # 模型及模型输入数据
    # model.add_graph = False
    val_iter = iter(get_dataloader(data_path='data/dev.tsv'))
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []
        pbar = tqdm(train_iter)
        pbar.set_description("[Train Epoch {}]".format(epoch))
        for src, trg, src_len, trg_len in pbar:
            trg, src = trg.to(device), src.to(device)
            model.zero_grad()
            output = model(src, trg)
            # trg = [batch size, trg len]
            # output = [batch size, trg len, output dim]
            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            epoch_loss.append(loss.item())
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            current_index = pbar.n
            if current_index % 100 == 0:
                writer.add_scalar("loss/train", loss.item(), current_index)
        loss_vals.append(np.mean(epoch_loss))

        model.eval()
        epoch_loss_eval = []
        pbar = tqdm(val_iter)
        pbar.set_description("[Eval Epoch {}]".format(epoch))
        for src, trg, src_len, trg_len in pbar:
            trg, src = trg.to(device), src.to(device)
            model.zero_grad()
            output = model(src, trg)
            # trg = [batch size, trg len]
            # output = [batch size, trg len, output dim]
            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)
            epoch_loss_eval.append(loss.item())
            pbar.set_postfix(loss=loss.item())
            current_index = pbar.n
            if current_index % 100 == 0:
                writer.add_scalar("loss/eval", loss.item(), current_index)
        loss_vals_eval.append(np.mean(epoch_loss_eval))
    writer.close()
    torch.save(model.state_dict(), f'save_model/model_{model_type}.pt')

    l1, = plt.plot(np.linspace(1, n_epochs, n_epochs).astype(int), loss_vals)
    l2, = plt.plot(np.linspace(1, n_epochs, n_epochs).astype(int), loss_vals_eval)
    plt.legend(handles=[l1, l2], labels=['Train loss', 'Eval loss'], loc='best')
