import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import hamming_loss, classification_report
import torchkeras
from model.sim_cse import SimCSEBert
from dataloader import get_dataloader
from torch.nn import functional as F


class SimCSEStepRunner(torchkeras.StepRunner):
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None, optimizer=None,
                 lr_scheduler=None):
        super().__init__(net, loss_fn, accelerator, stage, metrics_dict, optimizer, lr_scheduler)

    def compute_loss(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        with self.accelerator.autocast():
            preds = self.net(input_ids, attention_mask)
            loss = self.loss_fn(preds)
        return loss, None, None


def simcse_unsup_loss(y_pred):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size, 768]
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0])
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0]) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss


if __name__ == '__main__':
    save_dir = './saved_model/model.pt'
    epochs = 5
    bert_path = '../bert-base-chinese'
    train_data_path = './data/train_update_1.csv'
    dev_data_path = './data/dev.csv'
    batch_size = 6
    max_len = 512
    warmup_proportion = 0.1
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = simcse_unsup_loss
    model = SimCSEBert.from_pretrained(bert_path, method='pooler')
    dataset_type = 'sim_cse_train'
    train_dataloader = get_dataloader(data_path=train_data_path,
                                      tokenizer_path=bert_path,
                                      batch_size=batch_size,
                                      max_len=max_len,
                                      dataset_type=dataset_type)

    torchkeras.KerasModel.StepRunner = SimCSEStepRunner
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    model = torchkeras.KerasModel(model,
                                  loss_fn=loss_fn,
                                  optimizer=optimizer,
                                  lr_scheduler=scheduler
                                  )
    # for batch in dev_dataloader:
    #     tokens_ids, mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
    #     summary(model, input_data=[tokens_ids, mask])
    #     break
    from torchkeras.kerascallbacks import TensorBoardCallback

    dfhistory = model.fit(train_data=train_dataloader,
                          epochs=epochs,
                          patience=5,
                          ckpt_path=save_dir,
                          # callbacks=[TensorBoardCallback(save_dir='runs',
                          #                                model_name='mnist_cnn', log_weight=True, log_weight_freq=5)]
                          )
