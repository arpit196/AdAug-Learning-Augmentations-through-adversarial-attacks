import torchvision
import kornia
import os
from collections import OrderedDict
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import kornia
from torchmetrics.functional import accuracy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torchvision.models as models

class ResNet18Classifier(pl.LightningModule):
    def __init__(self, classes):
        super().__init__()
        self.model = models.resnet18(pretrained=False, num_classes=classes)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = nn.Identity()
        #self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.model(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        z = self(x)
        loss = F.cross_entropy(z, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="/content/drive/MyDrive/cifar10thesisrandaugment",
    filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import torchvision.transforms as T

dataset = CIFAR10(os.getcwd(), download=True, transform=transform)
train, val = random_split(dataset, [45000, 5000])
classifier = ResNet18Classifier(num_classes=10)
trainer = pl.Trainer(callbacks=[checkpoint_callback],accelerator='gpu', devices=1)
trainer.fit(classifier, DataLoader(train), DataLoader(val))

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import torchvision.transforms as T
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = CIFAR10(os.getcwd(), download=True, transform=transform_train)
train, val = random_split(dataset, [45000, 5000])
classifier = ResNet18Classifier(classes=10)
trainer = pl.Trainer(callbacks=[checkpoint_callback],accelerator='gpu', devices=1)
trainer.fit(classifier, DataLoader(train), DataLoader(val))