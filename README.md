# AdAug-Learning-Augmentations-through-adversarial-attacks
In this project, we propose an adversarial attack framework that learns an optimal configurations of augmentations for training deep learning based image classification model to improve model generalization and robustness. The adversarial attack framework learns the augmentation parameters in such a way that the cross entropy loss of the model is maximized as illustrated in figure for the rotation augmentation. This helps in finding configurations of augmentations on which the errors commited by the models are large and hence are the novel ones for the model and the most useful in improving model generalization. Our differentiable rotation augmentations outperforms random rotations by 1.7% on CIFAR10 and by 1.2% on CIFAR100 while our differentiable patch gaussian augmentation outperforms random patch gaussian by almost 5% on CIFAR10 and 2% on SVHN. Moreover, we show that the models trained through our differentiable rotation outperform even more diverse augmentations such as RandAugment and AutoAugment in the case of CIFAR-10 and CIFAR-100 datasets.

![alt text](https://i.ibb.co/YkGz1j7/Differentiable-Rotation-1.jpg)


![alt text](https://i.ibb.co/PrnF6gY/Adversarial-Results.png)

The above table compares the test accuracy of the different random, policy-based, and our differentiable learnable augmentation on the four datasets. The augmentations have been abbreviated as follows RR: Random Rotations, DR: Differentiable rotations, RPG: Random Patch Gaussian, DPG: Differentiable Patch Gaussian, RA:
RandAugment, and AA: AutoAugment. The differentiable ones are the ones learnt by our adversarial attack policy
