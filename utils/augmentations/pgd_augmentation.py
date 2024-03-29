import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
def pgd_attack_cutout(image, epsilon, data_grad, focal_mask):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad
    # Create the perturbed image by adjusting each pixel of the input image
    xhat = differentiable_cutout(image,torch.sigmoid(focal_mask + sign_data_grad*epsilon))
    # Return the perturbed image
    return xhat

def pgd_attack_focal(image, epsilon, data_grad, mean, variance, focal_mask):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad
    # Create the perturbed image by adjusting each pixel of the input image
    xhat = differentiable_gaussian_noise_with_selectable_focal_region(image,mean,variance,torch.sigmoid(focal_mask + sign_data_grad*epsilon))
    # Return the perturbed image
    return xhat
    
def pgd_attack(image, epsilon, data_grad, affine):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad
    xhat = differentiable_crop_translation_scale(image,affine+sign_data_grad.repeat(image.shape[0],1,1)*epsilon)
    return xhat

def pgd_attack_rotation(image, epsilon, data_grad, rot):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad
    sign_data_grad = torch.tensor(sign_data_grad)
    sign_data_grad = sign_data_grad.to(device)
    xhat = differentiable_rotation2(image,rot + sign_data_grad.repeat(image.shape[0],1)*epsilon)
    return xhat

def pgd_attack_blur(image, epsilon, data_grad, variance):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    xhat = differentiable_gaussian_blur(image,variance+sign_data_grad*epsilon)
    return xhat