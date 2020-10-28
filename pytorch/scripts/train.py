## Basic Python libraries
import sys
import os
sys.path.append(os.getcwd() + '/')

## Deep learning and array processing libraries
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader

## Metrics imports
import sklearn.metrics as metrics

## Inter-project imports
from config import configs
from core import attacks
from datasets import retrieve_dataset
from utils import train_network
from utils import parameters
from utils import BoldDriver
from utils import validation
from utils.recorder import TrainingRecorder

parser = argparse.ArgumentParser(description='Training a CNN with custom BN layer')
parser.add_argument('--dataset', '--data', '-d' default='cifar10', type=str, metavar='DATA', help='name of data set')
parser.add_argument('--name', default='', type=str, metavar='NAME', help='name of experiment')
parser.add_argument('--image_dir', '-I', default='', type=str, metavar='ID', help='location of image data')
parser.add_argument('--network_dir', '-N', default='', type=str, metavar='ND', help='location of network data')
parser.add_argument('--data_dir', '-D', default='', type=str, metavar='DD', help='location of parsed data')
parser.add_argument('--results_dir', '-R', default='', type=str, metavar='RD', help='location of results')
parser.add_argument('--network', '-n', default='resnet18_modified', type=str, metavar='N', choices=NETWORKS, help='network architecture')
parser.add_argument('--batch_size', '-b', default=128, type=int, metavar='BS', help='Batch size')
parser.add_argument('--learning_rate', '-l', default=0.1, type=float, metavar='LR', help='learning rate')
parser.add_argument('--weight_decay', '-w', default=0.0001, type=float, metavar='WD', help='weight decay')
parser.add_argument('--momentum', '-m', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--num_epochs', '-e', default=80, type=int, metavar='NE', help='number of epochs to train for')
parser.add_argument('--scheduler', default='bold', type=str, metavar='S', help='what type of learning rate scheduler')
parser.add_argument('--continue_training', '--cont', default='False', type=str, metavar='CONT', choices=['True', 'False'], help='are we continuing training')
parser.add_argument('--load_start', '--start', default='False', type=str, metavar='LOAD', choices=['True', 'False'], help='are we starting with some initial weights')
parser.add_argument('--gpu', '-g', default='cuda:0', type=str, metavar='G', choices=['cuda:0', 'cuda:1'], help='gpu id (e.g. \'cuda:0\'')
parser.add_argument('--seed', default=None, type=str, metavar='S', help='set a seed for reproducability')
args = vars(parser.parse_args())

args['continue_training'] = (args['continue_training'] == 'True')
args['load_start'] = (args['load_start'] == 'True')
args['seed'] = None if args['seed'] is None else int(args['seed'])

if __name__ == '__main__':
    # Set a seed with my birth date if we want reproducibility
    if args['seed'] is not None:
        torch.manual_seed(args['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args['seed'])

    # Create the experiment name
    experiment = f'{args["dataset"]}_train' if args['name'] == '' else args['name']

    # Define the compute device (either GPU or CPU)
    compute_device = torch.device(args['gpu'] if torch.cuda.is_available() else 'cpu')

    # Ensure we don't accidentally overwrite anything by checking how many previous experiments share the same name
    if not args['continue_training']:
        directories = [name for name in os.listdir(os.path.abspath(args['network_dir'])) if os.path.isdir(f'{args["network_dir"]}{name}') and experiment in name]
        num = len(directories)
        experiment = f'{experiment}_{num}'
        del directories

    # Set up a parameters object for saving hyperparameters, etc.
    parameters = parameters.Parameters(experiment, 'train', **args)
    if args['continue_training']:
        with open(os.path.abspath(f'{args["network_dir"]}{experiment}_parameters.pkl'), 'rb') as f:
            parameters = pickle.load(f)

    # Create the data transforms for each respective set
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # Retrieve the datasets
    train_dataset, val_dataset, _ = retrieve_dataset(args['dataset'], args['image_dir'], train_transform, test_transform, test_equals_val=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    # Create the network, (potentially) load network state dictionary, and send the network to the compute device
    num_classes = train_dataset.num_classes()
    loader = retrieve_network(args['dataset'], args['network'])
    network = loader(num_classes=num_classes)
    if load_weights:
        network.load_state_dict(torch.load(os.path.abspath(f'{args["network_dir"]}{experiment}/{experiment}_initial_weights_adversarial.pth'), map_location='cpu'))
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
    if args['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1)
    elif args['scheduler'] == 'bold':
        scheduler = BoldDriver(optimizer, network, decrease_factor=0.5, increase_factor=1.2)

    # Create classification loss function
    classification_loss_func = nn.CrossEntropyLoss()

    # Create a recorder
    recorder = TrainingRecorder(args['results_dir'], args['network_dir'], parameters, new=(parameters.epoch == 0))

    # Begin training the neural network
    for epoch in range(parameters.epoch, parameters.num_epochs):
        true_classes = np.empty(0)
        predicted_classes = np.empty(0)

        running_loss = 0.0
        network.train()
        for batch_num, batch_sample in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Load in batch image data
            image_data = batch_sample[0]
            image_data.requires_grad = False

            # Load in batch label data
            label_data = batch_sample[1]
            label_data.requires_grad = False

            # Send image and label data to device
            image_data = image_data.to(compute_device)
            label_data = label_data.to(compute_device)

            # Forward pass and get the output predictions
            predictions, _ = network(image_data)

            # Pass through loss function and perform back propagation
            loss = classification_loss_func(predictions, label_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(f'Experiment: {parameters.experiment} -- Epoch: {epoch} -- Batch: {batch_num} -- Loss: {loss.item()}')

            # Record the actual and predicted labels for the instance
            true_classes = np.concatenate((true_classes, label_data.detach().cpu().numpy()))
            _, predictions = torch.max(predictions, 1)
            predicted_classes = np.concatenate((predicted_classes, predictions.detach().cpu().numpy()))

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()
            elif isinstance(scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, BoldDriver)):
                scheduler.step(running_loss)

        # Get the training accuracy
        loss = running_loss / (batch_num + 1)
        accuracy = metrics.accuracy_score(true_classes, predicted_classes)

        # Check the validation error after each training epoch
        print(f'Evaluating validation set (epoch {epoch}):')
        val_loss, val_accuracy = validation(network=network, dataloader=val_dataloader, compute_device=compute_device, 
                                                        experiment=parameters.experiment, results_directory=args['results_dir'], 
                                                        classification_loss_func=classification_loss_func)

        recorder.record(epoch, loss, val_loss, accuracy, val_accuracy)
        recorder.update(epoch, val_loss, network.state_dict(), optimizer.state_dict())
