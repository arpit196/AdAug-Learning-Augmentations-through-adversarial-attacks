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
import torchvision.models as models
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import StepLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
seed_everything(7)
from torchmetrics.functional import accuracy
import torchvision
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)
from torchvision.datasets import STL10
#from differentiable_augmentations import * 
from torchvision.datasets import CIFAR10

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, p, length):
        self.n_holes = n_holes
        self.length = length
        self.p = p
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        if(np.random.rand(1)>self.p):
          return img

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = np.random.rand(y2-y1,x2-x1)

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        
class CIFARDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        augment: str="n"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment=augment

        if(self.augment=="auto"):
          self.transform = transforms.Compose(
              [
                  torchvision.transforms.RandomCrop(32, padding=4),
                  torchvision.transforms.RandomHorizontalFlip(),
                  transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                  transforms.ToTensor(),
                  transforms.Normalize((0.491,0.482,0.446), (0.247,0.243,0.261)),
              ]
          )
        elif(self.augment=="rot"):
          self.transform = transforms.Compose(
              [
                  torchvision.transforms.RandomCrop(32, padding=4),
                  torchvision.transforms.RandomHorizontalFlip(),
                  torchvision.transforms.RandomRotation((-10,10)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.491,0.482,0.446), (0.247,0.243,0.261)),
              ]
          )
        elif(self.augment=="rand"):
          self.transform = transforms.Compose(
              [
                  torchvision.transforms.RandomCrop(32, padding=4),
                  torchvision.transforms.RandomHorizontalFlip(),
                  torchvision.transforms.RandAugment(2,14),
                  transforms.ToTensor(),
                  transforms.Normalize((0.491,0.482,0.446), (0.247,0.243,0.261)),
              ]
          )

        elif(self.augment=="standard"):
          self.transform = transforms.Compose(
              [
                  torchvision.transforms.RandomCrop(32, padding=4),
                  torchvision.transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize((0.491,0.482,0.446), (0.247,0.243,0.261)),
              ]
          )
        
        elif(self.augment=="gaussian"):
          self.transform = transforms.Compose(
              [
                  torchvision.transforms.RandomCrop(32, padding=4),
                  torchvision.transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize((0.491,0.482,0.446), (0.247,0.243,0.261)),
                  AddGaussianNoise(mean=0,std=1.0)
              ]
          )
        
        elif(self.augment=="cutout"):
          self.transform = transforms.Compose(
              [
                  torchvision.transforms.RandomCrop(32, padding=4),
                  torchvision.transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize((0.491,0.482,0.446), (0.247,0.243,0.261)),
                  Cutout(length=8,n_holes=1,p=0.4)
              ]
          )
        
        elif(self.augment=="combined"):
          self.transform = transforms.Compose(
              [
                  torchvision.transforms.RandomCrop(32, padding=4),
                  torchvision.transforms.RandomHorizontalFlip(),
                  torchvision.transforms.RandomRotation((-10,10)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.491,0.482,0.446), (0.247,0.243,0.261)),
                  Cutout(length=8,n_holes=1,p=0.4)
              ]
          )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                torchvision.transforms.RandomRotation((-10,10)),
                transforms.Normalize((0.491,0.482,0.446), (0.247,0.243,0.261)),
                Cutout(length=8,n_holes=1,p=0.4)
            ]
        )

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = CIFAR10(self.data_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)