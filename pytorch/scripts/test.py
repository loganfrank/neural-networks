## Basic Python libraries
import sys
import os
import pickle
sys.path.append(os.getcwd() + '/')

## Deep learning and array processing libraries
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 

## Inter-project imports
from config import configs
from utils import evaluate_network
from utils import parameters
from datasets.datasets import retrieve_transform
from datasets.datasets import retrieve_dataset
from networks import retrieve_network

if __name__ == '__main__':

    dataset = input('what dataset: ')
    args = configs.get_args(dataset)
    num = int(input('what number: '))

    experiment = f'{dataset}_vanilla_{num}'

    # Define the compute device (either GPU or CPU)
    compute_device = torch.device(args['gpu'] if torch.cuda.is_available() else 'cpu')

    # Flag for if we want to load the network and continue training or fine tune
    load_network_flag = False
    load_weights = False

    if not load_network_flag:
        directories = [name for name in os.listdir(os.path.abspath(args['network_dir'])) if os.path.isdir(f'{args["network_dir"]}{name}') and experiment in name]
        num = len(directories)
        experiment = f'{experiment}_{num}'
        del directories

    # Set up a parameters object for saving hyperparameters, etc.
	parameters = parameters.Parameters(experiment, 'train', **args)
	with open(os.path.abspath(f'{args["network_dir"]}{experiment}_parameters.pkl'), 'rb') as f:
			parameters = pickle.load(f)

    # Create the data transforms for each respective set
    train_transform, test_transform = retrieve_transform(dataset)

    # Retrieve the datasets
    _, val_dataset, test_dataset = retrieve_dataset(dataset, args['image_dir'], train_transform, test_transform)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Create the network, (potentially) load network state dictionary, and send the network to the compute device
    num_classes = test_dataset.num_classes()
    network = retrieve_network(dataset, args['network'], num_classes)
    network.load_state_dict(torch.load(os.path.abspath(f'{args["network_dir"]}{experiment}/{experiment}.pth'), map_location='cpu'))    network = network.to(compute_device)
    network.eval()

    # Send to GPU
    network = network.to(compute_device)

    # Call a helper function to evaluate the neural network for validation and test sets
    print('Evaluating validation set:')
    evaluate_network.validate_network(network=network, dataloader=val_dataloader, compute_device=compute_device, 
                                        experiment=experiment, results_directory=args['results_dir'], classification_loss_func=nn.CrossEntropyLoss(), save=True)
    print('Evaluating test set:')
    evaluate_network.test_network(network=network, dataloader=test_dataloader, compute_device=compute_device, 
                                        experiment=experiment, results_directory=args['results_dir'], save=True)