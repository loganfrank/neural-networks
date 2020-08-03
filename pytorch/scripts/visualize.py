## Basic Python libraries
import sys
import os
import argparse
import yaml
import pickle
sys.path.append(os.getcwd() + '/')

## Deep learning and array processing libraries
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 

## Visualization imports
from PIL import Image

## Inter-project imports
from config import configs
from utils import evaluate_network
from utils import parameters
from datasets.datasets import retrieve_transform
from datasets.datasets import retrieve_dataset
from networks.networks import retrieve_network

## Set seeds
torch.manual_seed(42)

if __name__ == '__main__':
    dataset = input('what dataset: ')
    args = configs.get_args(dataset)
    num = int(input('what number: '))

    experiment = f'{dataset}_vanilla_{num}'

    # Define the compute device (either GPU or CPU)
    compute_device = torch.device(args['gpu'] if torch.cuda.is_available() else 'cpu')

    # Load in parameters from training
    with open(f'{args["network_dir"]}{experiment}/{experiment}_parameters.pkl', 'rb') as f:
        parameters = pickle.load(f)

    # Create the data transforms for each respective set
    train_transform, test_transform = retrieve_transform(dataset, visualize=True)

    # Retrieve the datasets
    train_dataset, val_dataset, test_dataset = retrieve_dataset(dataset, args['image_dir'], train_transform, test_transform, visualize=True)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Flag as to whether or not we should save the images
    save_images = True
    if save_images:
        if not os.path.exists(os.path.abspath(f'{args["results_dir"]}{experiment}/{experiment}/images')): 
            os.makedirs(os.path.abspath(f'{args["results_dir"]}{experiment}/{experiment}/images'))

    # Select a DataLoader
    dataloader = train_dataloader

    for index, batch in enumerate(dataloader):
        if save_images:
            image, label = batch
            image = image.numpy().squeeze()
            image = Image.fromarray(image)
            if dataset == 'cifar10':
                label = dataloader.dataset.classes_names[label.item()]
            image.save(os.path.abspath(f'{args["results_dir"]}{experiment}/{experiment}/images/{label}_{index}.png'))