import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from .utils.differantiableAugmentations import differentiable_rotation
class DifferentiableRotationModule(pl.LightningModule):
    def __init__(self,augment,classifier,epsilon=0.8,lr=0.1):
          super().__init__()
          self.classifier = classifier
          self.rot = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)  #Learnable differentiable rotation parameter
          self.augment=augment
          self.epsilon=epsilon
          self.affine = torch.nn.Parameter(torch.Tensor([[1.0,0.0,0.0],[0.0,1.0,0.0]]),requires_grad=True)
          self.affine_grid = torch.nn.Parameter(torch.Tensor.repeat(self.affine,[256,1,1]), requires_grad=True)
          self.lr=lr
          #self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.classifier(x)
        embedding = F.log_softmax(embedding,dim=1)
        return embedding
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, y = x.to(device), y.to(device)
        
        #select a subset to perform the augmentation
        subset = int(4*x[0].shape[0]/5)
        x_aug, y_aug = x[subset:], y[subset:]
        x_prev = x[0:subset]
        xhat=x
        if(self.augment=="rot"):
          xhat = differentiable_rotation(x_aug,self.rot.repeat(x_aug.shape[0], 1))  #augment the subset using the differentiable rotation
        else:
          xhat = differentiable_crop_translation_scale(x_aug,torch.Tensor.repeat(self.affine,[x_aug.shape[0],1,1]))

        yreal = self.classifier(xhat)
        yreal = yreal.to(device)

        loss_D = F.nll_loss(yreal, y_aug)#- torch.sum(self.variance)# + F.binary_cross_entropy(yreal, valid))/2.0
        tqdm_dict = {"d_loss": loss_D}
        loss_D.backward()
        self.optimizer.step()
        
        #Clipping the rotation angle to be within 20 degrees
        if(self.augment=="rot"):
          grad = self.rot.grad.data
          if(grad*self.epsilon>20.0):
            grad=20.0/self.epsilon
          elif(grad*self.epsilon<-20.0):
            grad=-20.0/self.epsilon
          
          perturbation = fgsm_attack_rotation(x_aug, self.epsilon, grad, self.rot)
        
        else:
          grad = self.affine.grad.data
          perturbation = fgsm_attack(x_aug, self.epsilon, grad, self.affine)

        x = torch.cat([x_prev,perturbation],0)
        yfake = self.classifier(x) 
        loss_D1 = F.nll_loss(yfake, y)
        #print(torch.argmax(yfake)) 
        self.log("train_loss", loss_D1)
        output = OrderedDict({"loss": loss_D1, "progress_bar": tqdm_dict, "log": tqdm_dict,"loss_D": loss_D})
        return output

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(
          self.parameters(),
          lr=self.lr,
          momentum=0.9,
          weight_decay=5e-4,
          )
        steps_per_epoch = 50000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                self.optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}