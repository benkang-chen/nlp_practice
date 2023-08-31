import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torchkeras
from text_classification.multil_label.model.TextRCNN_Bert import TextRCNN_Bert
from text_classification.multil_label.dataloader import get_dataloader


class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):
        tokens_ids, mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        # loss
        with self.accelerator.autocast():
            preds = self.net(tokens_ids, mask)
            loss = self.loss_fn(preds, labels)

        # backward()
        if self.stage == "train" and self.optimizer is not None:
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_loss = self.accelerator.gather(loss).sum()
        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)

        # losses (or plain metrics that can be averaged)
        step_losses = {self.stage + "_loss": all_loss.item()}

        # metrics (stateful metrics)
        step_metrics = {self.stage + "_" + name: metric_fn(all_preds, all_labels).item()
                        for name, metric_fn in self.metrics_dict.items()}

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses, step_metrics


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
    save_dir = './saved_model/model.pt'
    epochs = 5
    bert_path = '../../bert-base-chinese'
    train_data_path = 'data/multi-classification-train.txt'
    dev_data_path = 'data/multi-classification-test.txt'
    batch_size = 6
    max_len = 256
    warmup_proportion = 0.1
    model = TextRCNN_Bert.from_pretrained(bert_path)

    train_dataloader = get_dataloader(data_path=train_data_path,
                                      tokenizer_path=bert_path,
                                      batch_size=batch_size,
                                      max_len=max_len)
    dev_dataloader = get_dataloader(data_path=dev_data_path,
                                    tokenizer_path=bert_path,
                                    batch_size=batch_size,
                                    max_len=max_len,
                                    shuffle=False)

    torchkeras.KerasModel.StepRunner = StepRunner
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    # loss_fn = torch.nn.MultiLabelSoftMarginLoss()
    model = torchkeras.KerasModel(model,
                                  loss_fn=cal_loss,
                                  optimizer=optimizer,
                                  lr_scheduler=scheduler
                                  )
    # for batch in dev_dataloader:
    #     tokens_ids, mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
    #     summary(model, input_data=[tokens_ids, mask])
    #     break

    dfhistory = model.fit(train_data=train_dataloader,
                          val_data=dev_dataloader,
                          epochs=epochs,
                          patience=5,
                          ckpt_path=save_dir,
                          # callbacks=[TensorBoardCallback(save_dir='runs',
                          #                                model_name='mnist_cnn', log_weight=True, log_weight_freq=5)]
                          )
