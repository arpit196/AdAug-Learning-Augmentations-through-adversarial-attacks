import numpy as np
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
def differentiable_rotation(image_batch, angle):
    image_batch = image_batch.expand(image_batch.shape[0],3,32,32)
    angle =  angle.squeeze()

    # define the rotation center
    center = torch.ones(image_batch.shape[0], 2)
    center = center.to(device)
    center[..., 0] = image_batch.shape[3] / 2  # x
    center[..., 1] = image_batch.shape[2] / 2  # y

    # define the scale factor
    scale = torch.ones(image_batch.shape[0],2)
    scale = scale.to(device)

    # compute the transformation matrix
    M = kornia.geometry.transform.get_rotation_matrix2d(center, angle, scale)

    # apply the transformation to original image

    _, _, h, w = image_batch.shape
    rotated_image_batch = kornia.geometry.transform.warp_affine(image_batch, M, dsize=(h, w))
    return rotated_image_batch

def differentiable_crop_translation_scale(image_batch, angle):
    grid = F.affine_grid(angle, (image_batch.shape[0], 3, 32, 32))
    grid = grid.to(device)
    image_batch = F.grid_sample(image_batch, grid)
    return image_batch

def differentiable_gaussian_blur(image_batch, sigma):
    gauss = kornia.filters.gaussian_blur2d(input = image_batch, kernel_size=(3, 3), sigma=(sigma, sigma))

    # blur the image
    image_batch = gauss

    # convert back to numpy
    #image_batch = kornia.tensor_to_image(image_batch.byte()[0])
    return image_batch

def differentiable_cutout(
    image_batch, focal_mask):
    b, c, himg, wimg = image_batch.shape
    #distr = Normal(loc=mean, scale=variance)
    focal_mask=torch.sigmoid(focal_mask)

    image_batch = image_batch*focal_mask#torch.cat(attended_noise,dim=0).to(device)
    image_batch.squeeze(0)
    return image_batch

#Patch Gaussian
def differentiable_gaussian_noise_with_selectable_focal_region(
    image_batch, mean, variance, focal_mask):
    b, c, himg, wimg = image_batch.shape
    distr = torch.randn((b,*focal_mask.shape))
    distr = distr.to(device)
    sample_noise = distr*mean+variance
    sample_noise = sample_noise.to(device)

    attended_noise = sample_noise * focal_mask
    attended_noise = attended_noise.to(device)

    image_batch = image_batch + attended_noise#torch.cat(attended_noise,dim=0).to(device)
    image_batch.squeeze(0)
    return image_batch