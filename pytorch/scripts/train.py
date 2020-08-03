## Basic Python libraries
import sys
import os
import pickle
sys.path.append(os.getcwd() + '/')

## Deep learning and array processing libraries
import numpy as np 
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms 

## Inter-project imports
from config import configs
from utils import train_network
from utils import parameters
from datasets.datasets import retrieve_transform
from datasets.datasets import retrieve_dataset
from networks import retrieve_network

if __name__ == '__main__':
    dataset = input('what dataset: ')
    args = configs.get_args(dataset)

    experiment = f'{dataset}_vanilla'

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
    if load_network_flag:
        with open(os.path.abspath(f'{args["network_dir"]}{experiment}_parameters.pkl'), 'rb') as f:
            parameters = pickle.load(f)

    # Create the data transforms for each respective set
    train_transform, test_transform, normalize, _ = retrieve_transform(dataset)

    # Retrieve the datasets
    train_dataset, val_dataset, test_dataset = retrieve_dataset(dataset, args['image_dir'], train_transform, test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Create the network, (potentially) load network state dictionary, and send the network to the compute device
    num_classes = train_dataset.num_classes()
    network = retrieve_network(dataset, args['network'], num_classes)
    if load_weights:
        network.load_state_dict(torch.load(os.path.abspath(f'{args["network_dir"]}{experiment}/{experiment}_initial_weights.pth'), map_location='cpu'))
    elif load_network_flag:
        network.load_state_dict(torch.load(os.path.abspath(f'{args["network_dir"]}{experiment}/{experiment}.pth'), map_location='cpu'))
    else:
        # Make the directory if it doesn't exist
        if not os.path.exists(os.path.abspath(f'{args["network_dir"]}{experiment}')): 
            os.makedirs(os.path.abspath(f'{args["network_dir"]}{experiment}'))

        torch.save(network.state_dict(), os.path.abspath(f'{args["network_dir"]}{experiment}/{experiment}_initial_weights.pth'))
    
    # Ensure all parameters allow for gradient descent
    for parameter in network.parameters():
        parameter.requires_grad = True

    # Send to GPU
    network = network.to(compute_device)

    # Create the optimizer and (potentially) load the optimizer state dictionary
    optimizer = optim.SGD(network.parameters(), lr=args['learning_rate'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    if load_network_flag:
        optimizer.load_state_dict(torch.load(os.path.abspath(f'{args["network_dir"]}{experiment}_optimizer.pth')))

    # Create a learning rate scheduler -- this will reduce the learning rate by a factor when learning becomes stagnant
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=0.1)

    # Send network and other parameters to a helper function for training the neural network
    train_network.train_network(network=network, optimizer=optimizer, scheduler=scheduler, 
                                        parameters=parameters, train_dataloader=train_dataloader, 
                                        val_dataloader=val_dataloader, compute_device=compute_device, 
                                        network_directory=args['network_dir'], results_directory=args['results_dir'])