## Basic Python imports
import sys
import os
import argparse
import yaml
import pickle
from PIL import Image
sys.path.append(os.getcwd() + '/')

## Deep learning and array processing imports
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn

## Visualization imports
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sn

## Metrics imports
import sklearn.metrics as metrics

def test_baseline_network(network, dataloader, compute_device, experiment, results_directory, save_results=False, return_all=False):
    # Instantiate two arrays to keep track of the ground truth label and network prediction for each instance
    num_instances = len(dataloader)
    true_classes = np.zeros(num_instances)
    predicted_classes = np.zeros(num_instances)

    network.eval()

    # Begin evaluating the neural network
    for batch_num, batch_sample in enumerate(dataloader):
        # Load in batch image data
        image_data = batch_sample[0]
        image_data.requires_grad = False

        # Load in batch label data
        label_data = batch_sample[1]
        label_data.requires_grad = False

        # Keep a copy of the label data on the cpu
        label_data_cpu = torch.clone(label_data)

        # Send image and label data to device
        image_data = image_data.to(compute_device)
        label_data = label_data.to(compute_device)

        # Forward pass and get the output predictions
        predictions = network(image_data)
        predictions = torch.squeeze(predictions)
        predictions = torch.argmax(predictions)

        # Record the actual and predicted labels for the instance
        true_classes[batch_num] = label_data.detach().cpu().numpy()
        predicted_classes[batch_num] = predictions.detach().cpu().numpy() 

    # Compute the general accuracy, F1 score, Cohen's Kappa score, and confusion matrix for the data and print each
    accuracy = metrics.accuracy_score(true_classes, predicted_classes)
    micro_f1 = metrics.f1_score(true_classes, predicted_classes, average='micro')
    macro_f1 = metrics.f1_score(true_classes, predicted_classes, average='macro')
    weighted_f1 = metrics.f1_score(true_classes, predicted_classes, average='weighted')
    cohens_kappa = metrics.cohen_kappa_score(true_classes, predicted_classes)
    confusion_matrix = metrics.confusion_matrix(true_classes, predicted_classes)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)

    print(confusion_matrix.diagonal())

    print(f'Accuracy: {accuracy : 0.4f}')
    print('Accuracy by class:')
    for class_name, class_accuracy in zip(dataloader.dataset.classes_unique, confusion_matrix.diagonal()):
        print(f'\tClass {class_name}: { class_accuracy : 0.4f}')
    print(f'Micro F1: {micro_f1 : 0.4f}') 
    print(f'Macro F1: {macro_f1 : 0.4f}') 
    print(f'Weighted F1: {weighted_f1 : 0.4f}') 
    print(f'Cohen\'s Kappa: {cohens_kappa : 0.4f}')
    print(f'Confusion Matrx: \n{np.round(confusion_matrix, decimals=4)}')

    # Save the resulting metrics to a file (if desired)
    if save_results:
        # Save the general accuracy, F1 score, and Cohen's Kappa score
        with open(os.path.abspath(f'{results_directory}{experiment}_results_test.txt'), 'w') as f:
            f.write(f'Accuracy: {accuracy : 0.4f} \n')
            f.write(f'Micro F1: {micro_f1 : 0.4f} \n')
            f.write(f'Macro F1: {macro_f1 : 0.4f} \n')
            f.write(f'Weighted F1: {weighted_f1 : 0.4f} \n')
            f.write(f'Cohen\'s Kappa: {cohens_kappa : 0.4f} \n\n')
            f.write(f'Confusion Matrx: \n{np.round(confusion_matrix, decimals=4)} \n')

        # Create and save the confusion matrix
        cm_df = pd.DataFrame(confusion_matrix, index=dataloader.dataset.classes_unique, columns=dataloader.dataset.classes_unique)
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        sn.heatmap(cm_df, cmap='YlGnBu', cbar=False, annot=True, ax=ax)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.set_title('Test Set Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        fig.savefig(os.path.abspath(f'{results_directory}{experiment}_confusion_matrix_test.png'))
    
    if return_all:
        return accuracy, micro_f1, macro_f1, weighted_f1, cohens_kappa

def validate_baseline_network(network, dataloader, compute_device, experiment, results_directory, classification_loss_func, get_confusion_matrix=False):
    # Instantiate two arrays to keep track of the ground truth label and network prediction for each instance
    num_instances = len(dataloader)
    true_classes = np.zeros(num_instances)
    predicted_classes = np.zeros(num_instances)

    network.eval()

    val_loss = 0.0

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
        predictions = network(image_data)

        # Accumulate the validation loss for each batch
        loss = classification_loss_func(predictions, label_data)
        val_loss += loss.item()

        # Get the flat prediction
        predictions = torch.argmax(predictions, dim=1)

        # Record the actual and predicted labels for the instance
        true_classes[batch_num] = label_data.detach().cpu().numpy()
        predicted_classes[batch_num] = predictions.detach().cpu().numpy() 

    # Compute the general accuracy, F1 score, Cohen's Kappa score, and confusion matrix for the data and print each
    accuracy = metrics.accuracy_score(true_classes, predicted_classes)
    micro_f1 = metrics.f1_score(true_classes, predicted_classes, average='micro')
    macro_f1 = metrics.f1_score(true_classes, predicted_classes, average='macro')
    weighted_f1 = metrics.f1_score(true_classes, predicted_classes, average='weighted')
    cohens_kappa = metrics.cohen_kappa_score(true_classes, predicted_classes)
    confusion_matrix = metrics.confusion_matrix(true_classes, predicted_classes)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)

    print(f'Accuracy: {accuracy : 0.4f}')
    print('Accuracy by class:')
    for class_name, class_accuracy in zip(dataloader.dataset.classes_unique, confusion_matrix.diagonal()):
        print(f'\tClass {class_name}: { class_accuracy : 0.4f}')
    print(f'Micro F1: {micro_f1 : 0.4f}')
    print(f'Macro F1: {macro_f1 : 0.4f}')
    print(f'Weighted F1: {weighted_f1 : 0.4f}')
    print(f'Cohen\'s Kappa: {cohens_kappa : 0.4f}')
    print(f'Confusion Matrx: \n{np.round(confusion_matrix, decimals=4)}')

    # Save the confusion matrix for the validation set (if desired)
    if get_confusion_matrix:
        # Create and save the confusion matrix
        cm_df = pd.DataFrame(confusion_matrix, index=dataloader.dataset.classes_unique, columns=dataloader.dataset.classes_unique)
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        sn.heatmap(cm_df, cmap='YlGnBu', cbar=False, annot=True, ax=ax)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.set_title('Validation Set Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        fig.savefig(os.path.abspath(f'{results_directory}{experiment}_confusion_matrix_val.png'))


    return (val_loss / (batch_num + 1)), -cohens_kappa, accuracy



