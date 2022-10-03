# AdAug-Learning-Augmentations-through-adversarial-attacks
In this project, we proposed an adversarial attack framework that learns an optimal configurations of augmentations for training deep convolutional neural network based image classification model to improve model generalization and robustness. The adversarial attack framework learns those configurations of augmentations that maximizes the cross entropy classification loss of the model as illustrated in the figure pf the cat below for the case of rotation augmentation. This helps in finding configurations of augmentations, such as the angles in case of rotations, on which the errors commited by the models are large and hence images augmented through these augmentations are the novel ones for the model and the most difficult, and hence are the most useful in improving model generalization. Our differentiable rotation augmentations outperforms models trained using random rotation augmentations by 1.7% on CIFAR10 and by 1.2% on CIFAR100 while our differentiable patch gaussian augmentation outperforms model trained using random patch gaussian by almost 5% on CIFAR10 and 2% on SVHN. Moreover, we show that the models trained through our differentiable rotation outperform even more diverse augmentations such as RandAugment and AutoAugment in the case of CIFAR-10 and CIFAR-100 datasets. Our adversarial attack framework for the case of rotation augmentation can be theoretically described using the following equation

$$\underset{\delta \theta}{max} \ \dfrac{1}{n} \ \sum_{i=1}^{n} L(A(X_{i},\theta + \delta \theta),y_{i})  \\ s.t. \ ||\delta \theta||_{2} < B.$$

where $$\delta \theta$$ is the amount by which the model perturbs the rotation angle to maximize the loss, $$L(.)$$ is the cross-entropy loss function, and $$A(.)$$ is the rotation augmentation. $$X$$ and $$y$$ are the images and their respective labels in the batch.

![alt text](https://i.ibb.co/YkGz1j7/Differentiable-Rotation-1.jpg)


![alt text](https://i.ibb.co/PrnF6gY/Adversarial-Results.png)

The above table compares the test accuracy of the different random, policy-based, and our differentiable learnable augmentation on the four datasets. The augmentations have been abbreviated as follows RR: Random Rotations, DR: Differentiable rotations, RPG: Random Patch Gaussian, DPG: Differentiable Patch Gaussian, RA:
RandAugment, and AA: AutoAugment. The differentiable ones are the ones learnt by our adversarial attack policy

