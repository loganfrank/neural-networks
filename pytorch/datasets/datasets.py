## Basic Python imports
import os

## PyTorch imports
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

## Image / Array imports
import numpy as np
import pandas as pd
from PIL import Image

## Inner-project imports
from datasets.cifar10 import CIFAR10
from datasets.imagenet import ImageNet
from utils.transforms import GammaJitter
from utils.transforms import Interpolate

def retrieve_transform(dataset, visualize=False):
    if dataset == 'cifar10':
        if not visualize:
            train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5), 
                                                  GammaJitter(low=0.9, high=1.1), 
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=CIFAR10.normalization_values[-1.0]['mean'], std=CIFAR10.normalization_values[-1.0]['std'])])  

            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=CIFAR10.normalization_values[-1.0]['mean'], std=CIFAR10.normalization_values[-1.0]['std'])])

        else:
            train_transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(mean=ImageNet.normalization_values[dataset][-1.0]['mean'], std=CIFAR10.normalization_values[dataset][-1.0]['std'])])  

            test_transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(mean=ImageNet.normalization_values[dataset][-1.0]['mean'], std=CIFAR10.normalization_values[dataset][-1.0]['std'])]) 
            
    else:
        raise('Not yet implemented')

    return train_transform, test_transform

def retrieve_dataset(dataset, image_directory, train_transform, test_transform, dataframe=None, visualize=False):
    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=image_directory, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=image_directory, train=False, transform=test_transform, download=True)
        train_dataset, val_dataset = CIFAR10.create_validation(train_dataset, test_dataset, visualize=visualize)
        test_dataset = CIFAR10.create_test(test_dataset, visualize=visualize)

    else:
        raise('Not yet implemented')
        train_dataset, test_dataset = ImageNet.create_train_and_test(dataset, dataframe, image_directory, train_transform, test_transform, visualize=visualize)
        val_dataset = ImageNet.create_validation(dataset, dataframe, image_directory, test_transform, visualize=visualize)

    return train_dataset, val_dataset, test_dataset