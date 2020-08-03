## Basic Python imports
import sys
import os

## Deep learning and array processing imports
import numpy as np 
import torch

## Inner-project Imports
from utils.recorder import EvaluateRecorder

def test_network(network, dataloader, compute_device, experiment, results_directory, save=False):
    # Instantiate two arrays to keep track of the ground truth label and network prediction for each instance
    num_instances = len(dataloader)
    true_classes = np.zeros(num_instances)
    predicted_classes = np.zeros(num_instances)
    features = []

    network.eval()

    with torch.no_grad():
        # Begin evaluating the neural network
        for batch_num, batch_sample in enumerate(dataloader):
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
            predictions, instance_features = network(image_data)
            predictions = torch.squeeze(predictions)
            predictions = torch.argmax(predictions)

            # Record the actual and predicted labels for the instance
            true_classes[batch_num] = label_data.detach().cpu().numpy()
            predicted_classes[batch_num] = predictions.detach().cpu().numpy() 
            features.append(instance_features.squeeze().detach().cpu().numpy())

    recorder = EvaluateRecorder(results_directory, experiment, 'test')
    recorder.record(true_classes, predicted_classes, dataloader.dataset.classes, save=save)

    return true_classes, predicted_classes, np.array(features)

def validate_network(network, dataloader, compute_device, experiment, results_directory, classification_loss_func, save=False):
    # Instantiate two arrays to keep track of the ground truth label and network prediction for each instance
    num_instances = len(dataloader)
    true_classes = np.zeros(num_instances)
    predicted_classes = np.zeros(num_instances)
    features = []

    network.eval()

    val_loss = 0.0

    with torch.no_grad():
        # Begin evaluating the neural network
        for batch_num, batch_sample in enumerate(dataloader):
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
            predictions, instance_features = network(image_data)

            # Accumulate the validation loss for each batch
            loss = classification_loss_func(predictions, label_data)
            val_loss += loss.item()

            # Get the flat prediction
            predictions = torch.argmax(predictions, dim=1)

            # Record the actual and predicted labels for the instance
            true_classes[batch_num] = label_data.detach().cpu().numpy()
            predicted_classes[batch_num] = predictions.detach().cpu().numpy() 
            features.append(instance_features.squeeze().detach().cpu().numpy())

    recorder = EvaluateRecorder(results_directory, experiment, 'val')
    accuracy = recorder.record(true_classes, predicted_classes, dataloader.dataset.classes, save=save)
    
    return (val_loss / (batch_num + 1)), accuracy, true_classes, predicted_classes, np.array(features)
