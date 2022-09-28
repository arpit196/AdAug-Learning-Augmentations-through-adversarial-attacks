import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from .differentiable_rotation_model import DifferentiableRotationModule
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
from torchvision.datasets import CIFAR10

flags = tf.app.flags

flags.DEFINE_string('augmentation', 'standard', 'The differentiable or random augmentation to apply')
flags.DEFINE_string('classes', 10, 'The number of classes in the dataset')
flags.DEFINE_string('epochs', 200, 'The number of epochs to run' 
                    'summaries.')
flags.DEFINE_integer('num_layers', 20, 'Number of weighted layers. Valid ' 
                     'values: 20, 32, 44, 56, 110')

def main(_):
    #set training checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="Cifar1018",
        filename="Rotation",
        save_top_k=1,
        mode="min",
    )
    datamodule = CIFARDataModule(augment=FLAGS.augmentation)
    
    #initialize the model trainer
    trainer = pl.Trainer(callbacks=[checkpoint_callback,LearningRateMonitor(logging_interval="step")],
        max_epochs=FLAGS.epochs,
        gpus=AVAIL_GPUS,
        logger=TensorBoardLogger("lightning_logs/", name="Cifar1018rotlarge"),
    )
    
    classifier = ResnetClassifier(classes=FLAGS.classes,lr=0.05,grad_clip=0.1)
    
    
    if(FLAGS.augmentation=="rotation"):
        model = DifferentiableRotation(classifier=classifier,augment="rot",epsilon=100.0,lr=0.1)
    elif(FLAGS.augmentation=="patch_guassian"):
        model = DifferentiablePatchGaussian(classifier=classifier,epsilon=1.0,lr=0.1)
    elif(FLAGS.augmentation=="combined"):
        model = DifferentiablePatchGaussian(classifier=classifier,epsilon=1.0,lr=0.1)
    

    trainer.fit(model, dm)
    trainer.test(classifier, dm)
    torch.save({'epoch': 100,
                'model_state_dict': classifier.state_dict()}, 
              'Cifar1018rotd.pth')

if __name__ == "__main__":
    main()