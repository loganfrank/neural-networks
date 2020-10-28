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
from utils.recorder import EvaluateRecorder
from datasets import retrieve_dataset
from networks import retrieve_network    

parser = argparse.ArgumentParser(description='Training a CNN with custom BN layer')
parser.add_argument('--dataset', '--data', '-d' default='cifar10', type=str, metavar='DATA', help='name of data set')
parser.add_argument('--name', default='', type=str, metavar='NAME', help='name of experiment')
parser.add_argument('--image_dir', '-I', default='', type=str, metavar='ID', help='location of image data')
parser.add_argument('--network_dir', '-N', default='', type=str, metavar='ND', help='location of network data')
parser.add_argument('--data_dir', '-D', default='', type=str, metavar='DD', help='location of parsed data')
parser.add_argument('--results_dir', '-R', default='', type=str, metavar='RD', help='location of results')
parser.add_argument('--seed', default=None, type=str, metavar='S', help='set a seed for reproducability')
args = vars(parser.parse_args())

args['seed'] = None if args['seed'] is None else int(args['seed'])

if __name__ == '__main__':
    # Set a seed with my birth date if we want reproducibility
    if args['seed'] is not None:
        torch.manual_seed(args['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args['seed'])

    experiment = f'{dataset}_visualize' if args['name'] == '' else args['name']

    # Create the data transforms for each respective set
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # Retrieve the datasets
    train_dataset, val_dataset, test_dataset = retrieve_dataset(args['dataset'], args['image_dir'], transform, transform, test_equals_val=True)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Begin visualizing the validation set
    for index, batch in enumerate(val_dataloader):
        if save_images:
            image, label = batch
            image = image.numpy().squeeze()
            image = Image.fromarray(image)

            if dataset == 'cifar10':
                label = dataloader.dataset.classes_names[label.item()]

            # TODO Do some matplotlib plotting

            image.save(os.path.abspath(f'{args["results_dir"]}{experiment}/{experiment}/images/{label}_{index}.png'))