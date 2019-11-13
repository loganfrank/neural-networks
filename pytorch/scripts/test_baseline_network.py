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
import torchvision
import torchvision.transforms as transforms 

## Inter-project imports
from utils import train_network
from utils import parameters
from utils import datasets
from utils import evaluate_network
from networks.basic import two_layer_cnn
from networks.basic import three_layer_cnn
from networks.basic import one_layer_nn
from networks.basic import two_layer_nn
from networks.resnet18 import resnet18

args = type('', (), {})()
if sys.platform == 'win32':
    args.config = '/Users/frank.580/Desktop/agriculture/code/cse-fabe/config/logan_pc.yaml'
elif sys.platform == 'darwin':
    args.config = '/Users/loganfrank/Desktop/research/agriculture/code/cse-fabe/config/logan_mac.yaml'
elif sys.platform == 'linux':
    pass
args.experiment = 'test_imbalanced'


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

# Load in parameters from training
with open(f'{pv_network_directory}{args.experiment}_parameters.pkl', 'rb') as f:
    parameters = pickle.load(f)

# Load pandas dataframe
dataframe = pd.read_pickle(os.path.abspath(pv_data_directory + 'cse_fabe.pkl'))

# Create the data transforms for evaluating
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Load testing dataset and dataloader
test_dataset = datasets.dataset(image_root_directory=pv_image_directory, dataframe=dataframe, transform=test_transform, phase='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load validation dataset and dataloader
val_dataset = datasets.dataset(image_root_directory=pv_image_directory, dataframe=dataframe, transform=test_transform, phase='val')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Create the network, load network state dictionary, and send the network to the compute device
num_classes = len(test_dataset.classes_unique)
network = resnet18(num_classes)
network.load_state_dict(torch.load(os.path.abspath(f'{pv_network_directory}{parameters.experiment}.pth'), map_location='cpu'))
network = network.to(compute_device)

# Call a helper function to evaluate the neural network for validation and test sets
print('Evaluating validation set:')
evaluate_network.validate_baseline_network(network=network, dataloader=val_dataloader, compute_device=compute_device, 
                                                        experiment=parameters.experiment, results_directory=pv_results_directory, 
                                                        classification_loss_func=nn.CrossEntropyLoss(), get_confusion_matrix=True)

print('Evaluating test set:')
evaluate_network.test_baseline_network(network=network, dataloader=test_dataloader, compute_device=compute_device, 
                                    experiment=parameters.experiment, results_directory=pv_results_directory, save_results=True)