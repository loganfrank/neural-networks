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

def retrieve_dataset(dataset, image_directory, train_transform, test_transform, test_equals_val=True):
    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=image_directory, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=image_directory, train=False, transform=test_transform, download=True)

        # Are we using the test set as the test and validation set or should we create a pseudo-validation set from the train set?
        if not test_equals_val:
            train_dataset, val_dataset = CIFAR10.create_validation(train_dataset, test_dataset)
            test_dataset = CIFAR10.create_test(test_dataset)
        else:
            train_dataset.targets = torch.tensor(train_dataset.targets)
            test_dataset.targets = torch.tensor(test_dataset.targets)
            train_dataset = CIFAR10.create_train(train_dataset)
            val_dataset = CIFAR10.create_test(test_dataset, phase='val')
            test_dataset = CIFAR10.create_test(test_dataset, phase='test')

    else:
        raise('Not yet implemented')
        train_dataset, test_dataset = ImageNet.create_train_and_test(dataset, dataframe, image_directory, train_transform, test_transform, visualize=visualize)
        val_dataset = ImageNet.create_validation(dataset, dataframe, image_directory, test_transform, visualize=visualize)

    return train_dataset, val_dataset, test_dataset