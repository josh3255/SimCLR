import sys
sys.path.append('../')

from config import get_args

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_dataset():
    cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=get_random_transform())
    # test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                    download=True, transform=get_random_transform())

    return cifar10_dataset

def get_cifar10_dataloader(args):
    cifar10_dataset = get_cifar10_dataset()
    cifar10_dataloader = DataLoader(cifar10_dataset, args.batch_size, shuffle=True)
    
    return cifar10_dataloader

def get_random_transform(size=32, s=1.0):
    crop_and_resize = transforms.RandomResizedCrop(size=(size, size))
    rotate = transforms.RandomChoice([
        transforms.RandomRotation(90),
        transforms.RandomRotation(180),
        transforms.RandomRotation(270)
    ])
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    gaussian_blur = transforms.GaussianBlur(kernel_size=size // 10, sigma=(0.1, 2.0))
    random_transform = transforms.RandomChoice([crop_and_resize, color_distort, gaussian_blur, rotate])
    to_tensor = transforms.ToTensor()

    def transform_fn(image):
        transformed_images = [to_tensor(random_transform(image)) for _ in range(2)]
        return transformed_images

    return transform_fn
