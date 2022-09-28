#!pip install -r /content/lightning-hydra-template/requirements.txt
import kornia
import os
from collections import OrderedDict
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
import torchvision.models as models
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import StepLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
import torchvision
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)
from torchvision.datasets import STL10
from torchvision.datasets import CIFAR10

class ResnetClassifier(pl.LightningModule):
    def __init__(self, classes=10, lr=0.05, grad_clip=0.1):
        super().__init__()
        self.model = models.resnet18(pretrained=False, num_classes=classes)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = nn.Identity()
        self.grad_clip = grad_clip
        self.save_hyperparameters()
        self.loss=0

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.model(x)
        embedding = F.log_softmax(embedding, dim=1)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        z = self(x)
        loss = F.nll_loss(z, y)
        #nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.loss=loss

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
      optimizer = torch.optim.SGD(
          self.parameters(),
          lr=self.hparams.lr,
          momentum=0.9,
          weight_decay=5e-4,
      )
      steps_per_epoch = 50000 // BATCH_SIZE

      scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
      }
      return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}