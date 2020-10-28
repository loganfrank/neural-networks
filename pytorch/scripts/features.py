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
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 

## Other imports
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

## Inter-project imports
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
parser.add_argument('--network', '-n', default='resnet18_modified', type=str, metavar='N', choices=NETWORKS, help='network architecture')
parser.add_argument('--batch_size', '-b', default=128, type=int, metavar='BS', help='Batch size')
parser.add_argument('--gpu', '-g', default='cuda:0', type=str, metavar='G', choices=['cuda:0', 'cuda:1'], help='gpu id (e.g. \'cuda:0\'')
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

    experiment = f'{args["dataset"]}_features' if args['name'] == '' else args['name']

    # Define the compute device (either GPU or CPU)
    compute_device = torch.device(args['gpu'] if torch.cuda.is_available() else 'cpu')

    # Set up a parameters object for saving hyperparameters, etc.
    parameters = parameters.Parameters(experiment, 'test', **args)
    with open(os.path.abspath(f'{args["network_dir"]}{experiment}_parameters.pkl'), 'rb') as f:
        parameters = pickle.load(f)

    # Create the data transforms for each respective set
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # Retrieve the datasets
    _, val_dataset, test_dataset = retrieve_dataset(args['dataset'], args['image_dir'], transform, transform, test_equals_val=True)

    val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

    # Create the network, (potentially) load network state dictionary, and send the network to the compute device
    num_classes = val_dataset.num_classes()
    loader = retrieve_network(args['dataset'], args['network'])
    network = loader(num_classes=num_classes)
    network.load_state_dict(torch.load(os.path.abspath(f'{args["network_dir"]}{experiment}/{experiment}.pth'), map_location='cpu'))
    network.eval()

    # Send to GPU
    network = network.to(compute_device)

    # Get the batch size
    batch_size = args['batch_size']

    # VALIDATION
    # Instantiate two arrays to keep track of the ground truth label and network prediction for each instance
    num_instances = len(val_dataset)
    true_classes = np.zeros(num_instances)
    predicted_classes = np.zeros(num_instances)

    with torch.no_grad():
        # Begin evaluating the neural network
        for batch_num, batch_sample in enumerate(val_dataloader):
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
            predictions, batch_features, _ = network(image_data)

            # Get the flat prediction
            predictions = torch.argmax(predictions, dim=1)

            # Record the actual and predicted labels for the instance
            true_classes[ batch_num * batch_size : min( (batch_num + 1) * batch_size, num_instances) ] = label_data.detach().cpu().numpy()
            predicted_classes[ batch_num * batch_size : min( (batch_num + 1) * batch_size, num_instances) ] = predictions.detach().cpu().numpy() 
            features.append(batch_features.squeeze().detach().cpu().numpy())

        features = np.array(features)

    val_classes, val_predictions = true_classes.reshape((-1, 1)), predicted_classes.reshape((-1, 1))
    val = np.hstack((val_classes, val_predictions, features))

    # TEST
    # Instantiate two arrays to keep track of the ground truth label and network prediction for each instance
    num_instances = len(test_dataset)
    true_classes = np.zeros(num_instances)
    predicted_classes = np.zeros(num_instances)

    with torch.no_grad():
        # Begin evaluating the neural network
        for batch_num, batch_sample in enumerate(test_dataloader):
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
            predictions, batch_features, _ = network(image_data)

            # Get the flat prediction
            predictions = torch.argmax(predictions, dim=1)

            # Record the actual and predicted labels for the instance
            true_classes[ batch_num * batch_size : min( (batch_num + 1) * batch_size, num_instances) ] = label_data.detach().cpu().numpy()
            predicted_classes[ batch_num * batch_size : min( (batch_num + 1) * batch_size, num_instances) ] = predictions.detach().cpu().numpy()
            features.append(batch_features.squeeze().detach().cpu().numpy())

        features = np.array(features)
    
    test_classes, test_predictions = true_classes.reshape((-1, 1)), predicted_classes.reshape((-1, 1))
    test = np.hstack((test_classes, test_predictions, features))

    np.savetxt(f'{args["data_dir"]}{dataset}_val_features.csv', val, delimiter=',')
    np.savetxt(f'{args["data_dir"]}{dataset}_test_features.csv', test, delimiter=',')

    plt.close('all')

    # [10000, 2] Matrix of X,Y points
    tsne = TSNE(n_components=2, random_state=args['seed']).fit_transform(test_features)

    # initialize a matplotlib plot
    fig = plt.figure()
    tsne_x, tsne_y = tsne[:, 0], tsne[:, 1]

    # Normalize
    tsne_x = (tsne_x - np.min(tsne_x)) / (np.max(tsne_x) - np.min(tsne_x))
    tsne_y = (tsne_y - np.min(tsne_y)) / (np.max(tsne_y) - np.min(tsne_y))

    # Plot each class
    for index in range(num_classes):
        class_instances = np.squeeze(test_classes == index)
        plt.scatter(tsne_x[class_instances], tsne_y[class_instances], s=0.2, label=test_dataset.classes_names[index])

    plt.legend(loc='best', markerscale=6, fontsize='xx-small', ncol=2)
    plt.title(f'{experiment}')
    plt.savefig(f'{args["results_dir"]}tsne_{experiment}.png')
    plt.show()

    print('stop')