import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import hamming_loss, classification_report
import torchkeras
from model.bert_re import BertForRE
from dataloader import get_dataloader
from dataloader import start_tag, stop_tag, tag2idx, idx2tag, relation2idx, idx2relation


class REStepRunner(torchkeras.StepRunner):
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None, optimizer=None,
                 lr_scheduler=None):
        super().__init__(net, loss_fn, accelerator, stage, metrics_dict, optimizer, lr_scheduler)

    def compute_loss(self, batch_data):
        input_ids = batch_data['input_ids']
        tag_ids = batch_data['tag_ids']
        attention_mask = batch_data['attention_mask']
        sub_mask = batch_data['sub_mask']
        obj_mask = batch_data['obj_mask']
        labels = batch_data['labels']
        real_lens = batch_data['real_lens']
        with self.accelerator.autocast():
            loss = self.net(input_ids, attention_mask, tag_ids, sub_mask, obj_mask, labels, real_lens)
            # loss = self.loss_fn(loss)
        return loss, None, None


# def loss_f(loss):
#     """
#     loss 已经在模型计算
#     """
#     return loss


if __name__ == '__main__':
    save_dir = './saved_model/model.pt'
    epochs = 5
    bert_path = '../bert-base-chinese'
    train_data_path = './data/train_data.json'
    batch_size = 32
    max_len = 512
    warmup_proportion = 0.1
    loss_fn = None
    model = BertForRE.from_pretrained(bert_path, start_tag, stop_tag, tag2idx, idx2tag, relation2idx, idx2relation)
    train_dataloader = get_dataloader(data_path=train_data_path,
                                      tokenizer_path=bert_path,
                                      batch_size=batch_size,
                                      max_len=max_len)

    torchkeras.KerasModel.StepRunner = REStepRunner
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
