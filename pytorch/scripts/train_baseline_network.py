## Basic Python libraries
import sys
import os
import argparse
import yaml
import pickle
sys.path.append(os.path.abspath(os.getcwd() + '/pytorch/'))

## Deep learning and array processing libraries
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 

## Inter-project imports
from utils import train_network
from utils import evaluate_network
from utils import datasets
from utils import parameters
from utils.transforms import RotationTransform
from utils.transforms import GammaJitter
from utils.transforms import RandomScale
from networks.basic import two_layer_nn

args = type('', (), {})()
if sys.platform == 'win32':
    args.config = '/Users/frank.580/Desktop/agriculture/code/cse-fabe/config/logan_pc.yaml'
elif sys.platform == 'darwin':
    args.config = '/Users/loganfrank/Desktop/research/agriculture/code/cse-fabe/config/logan_mac.yaml'
elif sys.platform == 'linux':
    pass
args.experiment = 'test_imbalanced'
args.batch_size = 32
args.learning_rate = 0.001
args.early_stopping = False
args.num_epochs = 200

# Open the yaml config file
try:
    with open(os.path.abspath(args.config)) as config_file: 
        config = yaml.safe_load(config_file)

        # Location of root directory of all images
        pv_image_directory = config['Paths']['pv_image_directory']

        # Location of network parameters (network state dicts, etc.)
        pv_network_directory = config['Paths']['pv_network_directory']

        # Location of parsed data (dataframes, etc.)
        pv_data_directory = config['Paths']['pv_data_directory']

        # Location of saved results from evaluation (confusion matrix, etc.)
        pv_results_directory = config['Paths']['pv_results_directory']

except:
    raise Exception('Error loading data from config file.')

# Define the compute device (either GPU or CPU)
compute_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Flag for if we want to load the network and continue training or fine tune
load_network_flag = False

# Set up a parameters object for saving hyperparameters, etc.
parameters = parameters.Parameters()
if load_network_flag:
    with open(os.path.abspath(f'{pv_network_directory}{args.experiment}_parameters.pkl'), 'rb') as f:
        parameters = pickle.load(f)
else:
    parameters.experiment = args.experiment
    parameters.batch_size = args.batch_size
    parameters.learning_rate = args.learning_rate
    parameters.early_stopping = args.early_stopping
    parameters.num_epochs = args.num_epochs

# Load pandas dataframe
dataframe = pd.read_pickle(os.path.abspath(pv_data_directory + 'cse_fabe.pkl'))

# Create the data transforms for each respective set
train_transform = transforms.Compose([RandomScale(low=1.0, high=1.1), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), 
                                    RotationTransform(angles=[0, 90, 180, 270]), GammaJitter(low=0.9, high=1.1), 
                                    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])                         
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Load training dataset and dataloader
train_dataset = datasets.dataset(image_root_directory=pv_image_directory, dataframe=dataframe,
                                    transform=train_transform, phase='train')
train_dataloader = DataLoader(train_dataset, batch_size=parameters.batch_size, shuffle=True)

# Load testing dataset and dataloader
test_dataset = datasets.dataset(image_root_directory=pv_image_directory, dataframe=dataframe,  
                                        transform=test_transform, phase='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load validation dataset and dataloader
val_dataset = datasets.dataset(image_root_directory=pv_image_directory, dataframe=dataframe, 
                                        transform=test_transform, phase='val')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Create the network, (potentially) load network state dictionary, and send the network to the compute device
num_classes = len(train_dataset.classes_unique)
network = two_layer_nn(256 * 256 * 3, num_classes)
if load_network_flag:
    network.load_state_dict(torch.load(os.path.abspath(f'{pv_network_directory}{parameters.experiment}.pth'), map_location='cpu'))
for parameter in network.parameters():
    parameter.requires_grad = True

# If we can use multiple GPUs, do so
if torch.cuda.device_count() > 1:
    print('Converting to DataParallel network')
    network = nn.DataParallel(network)
    parameters.parallel = True

# Send to GPU
network = network.to(compute_device)

# Create the optimizer and (potentially) load the optimizer state dictionary
optimizer = optim.SGD(network.parameters(), lr=parameters.learning_rate, momentum=parameters.momentum, weight_decay=parameters.weight_decay)
if load_network_flag:
    optimizer.load_state_dict(torch.load(os.path.abspath(f'{pv_network_directory}{parameters.experiment}_optimizer.pth')))

# Create a learning rate scheduler -- this will reduce the learning rate by a factor when learning becomes stagnant
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Send network and other parameters to a helper function for training the neural network
try:
    train_network.train_network(network=network, optimizer=optimizer, scheduler=scheduler, 
                                        parameters=parameters, train_dataloader=train_dataloader, 
                                        val_dataloader=val_dataloader, compute_device=compute_device, 
                                        network_directory=pv_network_directory, results_directory=pv_results_directory)
    print('Testing best network')
    network.load_state_dict(torch.load(os.path.abspath(f'{pv_network_directory}{parameters.experiment}.pth'), map_location='cpu'))
    network = network.to(compute_device)
    evaluate_network.test_baseline_network(network=network, dataloader=test_dataloader, compute_device=compute_device, 
                                        experiment=parameters.experiment, results_directory=pv_results_directory, save_results=True)
except:
    print('test_imbalanced failed')