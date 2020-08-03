## Basic Python imports
import sys
import os
import yaml
sys.path.append(os.getcwd() + '/')

## Other imports
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

## Inner-project imports
from datasets.datasets import retrieve_dataset
from core.dropout import ImageDropout
from utils import command_line_args

def get_mean_and_std(dataset):
    """
    Compute the mean and std value of dataset.
    Taken from github.com/facebookresearch/mixup-cifar10
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

if __name__ == '__main__':
    dataset = input('what dataset: ')
    args = configs.get_args(dataset)

    experiment = f'{dataset}_vanilla'

    if dataset == 'mnist' or dataset == 'fmnist':
        transform = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    train_dataset, _, _ = retrieve_dataset(dataset, image_directory, transform, transform)

    means = np.zeros((10, 3))
    std_devs = np.zeros((10, 3))
    for run in range(10):
        mean, std_dev = get_mean_and_std(train_dataset)
        means[run, :] = mean
        std_devs[run, :] = std_dev

    with open(os.path.abspath(f'{data_directory}normalization_values.txt'), 'w') as f:
        for mean in means:
            f.write(f'Mean: {mean.round(4)} \n')
        f.write(f'\nAvg-Mean: {(np.sum(means, axis=0) / len(means)).round(4)} \n')
        f.write('\n')

        for std_dev in std_devs:
            f.write(f'Std-Dev: {std_dev.round(4)} \n')
        f.write(f'\nAvg-Std-Dev: {(np.sum(std_devs, axis=0) / len(std_devs)).round(4)}\n')