import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
class DifferentiablePatchGaussianModule(pl.LightningModule):
    def __init__(self,classifier,lr=0.1,epsilon=1.0):
          super().__init__()
          self.classifier = classifier
          self.focal_mask = torch.nn.Parameter(torch.rand((3,32,32), requires_grad=True))
          self.mean = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
          self.variance = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
          self.lr=lr
          self.epsilon = epsilon
          #self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.classifier(x)
        embedding = F.softmax(embedding, dim=1)
        return embedding
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        
        subset = int(4*x[0].shape[0]/5)
        x_aug, y_aug = x[subset:], y[subset:]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, y = x.to(device), y.to(device)
        xhat = differentiable_gaussian_noise_with_selectable_focal_region(x_aug,self.mean,self.variance,self.focal_mask)

        yreal = self.classifier(xhat)
        yreal = yreal.to(device)

        loss_Dg = F.cross_entropy(yreal, y_aug) + torch.sum(self.focal_mask)#- torch.sum(self.variance)# + F.binary_cross_entropy(yreal, valid))/2.0
        tqdm_dict = {"d_loss": loss_Dg}
        

        gradm = torch.autograd.grad(loss_Dg, self.mean,
                                  retain_graph=True, create_graph=False)[0]
        gradv = torch.autograd.grad(loss_Dg, self.variance,
                                  retain_graph=True, create_graph=False)[0]
        gradf = torch.autograd.grad(loss_Dg, self.focal_mask,
                                  retain_graph=False, create_graph=False)[0]
        #print(self.rot+grad)
        perturbation = fgsm_attack_focal(x_aug, self.epsilon, gradf,self.mean+gradm,self.variance+gradv,self.focal_mask)
        
        x = torch.cat([x[0:subset],perturbation],0)
        yfake = self.classifier(x)
        loss_D1 = F.cross_entropy(yfake, y)
        print('{} seconds'.format(time.time() - t0)) 
        self.log("train_loss", loss_D1)
        output = OrderedDict({"loss": loss_D1, "progress_bar": tqdm_dict, "log": tqdm_dict,"loss_D": loss_Dg})
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
          self.parameters(),
          lr=self.lr,
          momentum=0.9,
          weight_decay=5e-4,
          )
        steps_per_epoch = 45000 // BATCH_SIZE
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