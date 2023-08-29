import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import torchkeras


def create_net():
    net = nn.Sequential()
    net.add_module("conv1", nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3))
    net.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
    net.add_module("conv2", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5))
    net.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
    net.add_module("dropout", nn.Dropout2d(p=0.1))
    net.add_module("adaptive_pool", nn.AdaptiveMaxPool2d((1, 1)))
    net.add_module("flatten", nn.Flatten())
    net.add_module("linear1", nn.Linear(64, 32))
    net.add_module("relu", nn.ReLU())
    net.add_module("linear2", nn.Linear(32, 10))
    return net


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

        self.correct = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.argmax(dim=-1)
        m = (preds == targets).sum()
        n = targets.shape[0]
        self.correct += m
        self.total += n

        return m / n

    def compute(self):
        return self.correct.float() / self.total

    def reset(self):
        self.correct -= self.correct
        self.total -= self.total


if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=transform)
    ds_val = torchvision.datasets.MNIST(root="mnist/", train=False, download=True, transform=transform)
    dl_train = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=128, shuffle=False, num_workers=2)
    net = create_net()
    model = torchkeras.KerasModel(net,
                                  loss_fn=nn.CrossEntropyLoss(),
                                  optimizer=torch.optim.Adam(net.parameters(), lr=0.001),
                                  metrics_dict={"acc": Accuracy()}
                                  )
    for features, labels in dl_train:
        break
    from torchkeras import summary

    summary(model, input_data=features)
    from torchkeras.kerascallbacks import TensorBoardCallback

    dfhistory = model.fit(train_data=dl_train,
                          val_data=dl_val,
                          epochs=2,
                          patience=5,
                          monitor="val_acc", mode="max",
                          ckpt_path='checkpoint.pt',
                          # callbacks=[TensorBoardCallback(save_dir='runs',
                          #                                model_name='mnist_cnn', log_weight=True, log_weight_freq=5)]
                          )
    print(dfhistory)