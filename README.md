# AdAug-Learning-Augmentations-through-adversarial-attacks
In this project, we propose an adversarial attack framework that learns an optimal configurations of augmentations for improving model generalization and robustness and to avoid the expensive search phase for optimization that is used in the policy optimization phase in AutoAugment. The adversarial attack framework learns the augmentation parameters in such a way that the cross entropy loss of the model is maximized as illustrated in figure for the rotation augmentation.  Our differentiable
rotation augmentations outperform random rotations by 1.7% on CIFAR10 and by 1.2% on CIFAR100 while our differentiable patch gaussian augmentation outperforms random patch gaussian by almost 5% on CIFAR10 and 2% on SVHN. Moreover, we show that the models trained through our differentiable rotation outperform even more diverse augmentations such as RandAugment and AutoAugment in the case of CIFAR-10 and CIFAR-100 datasets.

![alt text](https://i.ibb.co/YkGz1j7/Differentiable-Rotation-1.jpg)


![alt text](https://i.ibb.co/44Cb7QZ/Adv.png)
