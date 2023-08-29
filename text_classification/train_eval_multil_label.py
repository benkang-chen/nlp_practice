import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import hamming_loss, classification_report
from tqdm import tqdm
from model.TextRCNN_Bert import TextRCNN_Bert
from torch.utils.tensorboard import SummaryWriter

from text_classification.dataloader import get_dataloader


# train
def cal_loss(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred_pos[:, :1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), 1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), 1)
    neg_loss = torch.logsumexp(y_pred_neg, 1)
    pos_loss = torch.logsumexp(y_pred_pos, 1)
    loss = torch.mean(neg_loss + pos_loss)
    return loss


if __name__ == '__main__':

    save_dir = './saved_model'
    epochs = 5
    bert_path = '../bert-base-chinese'
    train_data_path = './data/multi-classification-train.txt'
    dev_data_path = './data/multi-classification-test.txt'
    batch_size = 64
    max_len = 256
    warmup_proportion = 0.1
    loss_type = "MultiLabelSoftMarginLoss"
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    model = TextRCNN_Bert.from_pretrained(bert_path)
    model.to(device)

    train_dataloader = get_dataloader(data_path=train_data_path,
                                      tokenizer_path=bert_path,
                                      batch_size=batch_size,
                                      max_len=max_len)
    dev_dataloader = get_dataloader(data_path=dev_data_path,
                                    tokenizer_path=bert_path,
                                    batch_size=batch_size,
                                    max_len=max_len,
                                    shuffle=False)
    # writer = SummaryWriter("./runs")
    # test_data = next(iter(traindataloader))

    total_steps = len(train_dataloader) * epochs
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    loss_vals = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        pbar = tqdm(train_dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        for batch in pbar:
            tokens_ids, mask, label = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch[
                'labels'].to(
                device)
            model.zero_grad()
            out = model(tokens_ids, mask)
            if loss_type == 'MultiLabelSoftMarginLoss':
                loss = criterion(out, label)
            else:
                loss = cal_loss(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            epoch_loss.append(loss.item())
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=loss.item())
        loss_vals.append(np.mean(epoch_loss))
    model.save_pretrained(save_dir)
    plt.plot(np.linspace(1, epochs, epochs).astype(int), loss_vals)

    # eval
    model = TextRCNN_Bert.from_pretrained(save_dir)
    model.to(device)
    model.eval()
    pred_y = np.empty((0, 65))
    true_y = np.empty((0, 65))
    with torch.no_grad():
        for batch in dev_dataloader:
            tokens_ids, mask, label = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch[
                'labels'].to(
                device)
            logits = model(tokens_ids, mask)
            pred = torch.where(logits > 0, 1, 0)
            pred = pred.cpu().numpy()
            label = label.cpu().numpy()
            pred_y = np.append(pred_y, pred, axis=0)
            true_y = np.append(true_y, label, axis=0)
    print(classification_report(true_y, pred_y, digits=4))
    h_loss = hamming_loss(true_y, pred_y)
    print(h_loss)
