import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import hamming_loss, classification_report
import torchkeras
from model.sentence_bert import SentenceBert
from dataloader import get_dataloader


class SentenceBertStepRunner(torchkeras.StepRunner):
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None, optimizer=None,
                 lr_scheduler=None):
        super().__init__(net, loss_fn, accelerator, stage, metrics_dict, optimizer, lr_scheduler)

    def compute_loss(self, batch):
        input_ids_1 = batch['input_ids_1']
        attention_mask_1 = batch['attention_mask_1']
        input_ids_2 = batch['input_ids_2']
        attention_mask_2 = batch['attention_mask_2']
        labels = batch['labels']
        with self.accelerator.autocast():
            if self.net.use_cosine_embedding_loss:
                embed_1, embed_2 = self.net(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                preds = torch.where(labels == 0, -1, labels)
                loss = self.loss_fn(embed_1, embed_2, labels)
            else:
                preds = self.net(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = self.loss_fn(preds, labels)
        return loss, labels, preds


if __name__ == '__main__':
    save_dir = './saved_model/model.pt'
    epochs = 5
    bert_path = '../bert-base-chinese'
    train_data_path = './data/train.csv'
    dev_data_path = './data/dev.csv'
    batch_size = 6
    max_len = 512
    warmup_proportion = 0.1
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.CosineEmbeddingLoss()
    use_cosine_embedding_loss = True if isinstance(loss_fn, nn.CosineEmbeddingLoss) else False
    model = SentenceBert.from_pretrained(bert_path, use_cosine_embedding_loss=use_cosine_embedding_loss)

    train_dataloader = get_dataloader(data_path=train_data_path,
                                      tokenizer_path=bert_path,
                                      batch_size=batch_size,
                                      max_len=max_len)
    dev_dataloader = get_dataloader(data_path=dev_data_path,
                                    tokenizer_path=bert_path,
                                    batch_size=batch_size,
                                    max_len=max_len,
                                    shuffle=False)

    torchkeras.KerasModel.StepRunner = SentenceBertStepRunner
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
                          val_data=dev_dataloader,
                          epochs=epochs,
                          patience=5,
                          ckpt_path=save_dir,
                          # callbacks=[TensorBoardCallback(save_dir='runs',
                          #                                model_name='mnist_cnn', log_weight=True, log_weight_freq=5)]
                          )
