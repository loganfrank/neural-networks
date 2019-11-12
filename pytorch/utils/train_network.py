## Basic Python imports
import sys
import os
import argparse
import yaml
import pickle
sys.path.append(os.getcwd() + '/')

## Deep learning and array processing imports
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn

## Inter-project imports
from utils import evaluate_network
from utils import functions

## Metrics imports
import sklearn.metrics as metrics

def train_network(network, optimizer, scheduler, parameters, train_dataloader, val_dataloader, compute_device, network_directory, results_directory):
    # Create classification loss function
    classification_loss_func = nn.CrossEntropyLoss()

    # Create a file to record the average loss values for each epoch
    if parameters.epoch == 0:
        with open(os.path.abspath(f'{results_directory}{parameters.experiment}_average_losses.txt'), 'w') as output_running_loss_file:
            output_running_loss_file.write(f'{parameters.experiment}\n')
            output_running_loss_file.write('(epoch number), (average training loss for given epoch number), (average validation loss for given epoch number) \n')

        with open(os.path.abspath(f'{results_directory}{parameters.experiment}_accuracies.txt'), 'w') as output_accuracy_file:
            output_accuracy_file.write(f'{parameters.experiment}\n')
            output_accuracy_file.write('(epoch number), (training accuracy for given epoch number), (validation accuracy for given epoch number) \n')

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
            image_data = torch.squeeze(image_data)
            image_data.requires_grad = False

            # Load in batch label data
            label_data = batch_sample[1]
            label_data = torch.squeeze(label_data) 
            label_data.requires_grad = False

            # Send image and label data to device
            image_data = image_data.to(compute_device)
            label_data = label_data.to(compute_device)

            # Forward pass and get the output predictions
            predictions = network(image_data)
            predictions = torch.squeeze(predictions)

            # Pass through loss function and perform back propagation
            loss = classification_loss_func(predictions, label_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(f'Experiment: {parameters.experiment} -- Epoch: {epoch} -- Batch: {batch_num} -- Loss: {loss.item()}')

            # Record the actual and predicted labels for the instance
            true_classes = np.append(true_classes, label_data.detach().cpu().numpy())
            _, predictions = torch.max(predictions, 1)
            predicted_classes = np.append(predicted_classes, predictions.detach().cpu().numpy())

        
        scheduler.step(running_loss)

        # Check the validation error after each training epoch
        print(f'Evaluating validation set (epoch {epoch}):')
        val_loss, _, val_accuracy = evaluate_network.validate_baseline_network(network=network, dataloader=val_dataloader, compute_device=compute_device, 
                                                            experiment=parameters.experiment, results_directory=results_directory, classification_loss_func=classification_loss_func)

        # Append the epoch number and the average loss for that epoch to the previously created average loss output file
        with open(os.path.abspath(f'{results_directory}{parameters.experiment}_average_losses.txt'), 'a+') as output_running_loss_file:
            output_running_loss_file.write(f'{epoch}, {running_loss / (batch_num + 1) : 0.4f}, {val_loss : 0.4f} \n')
        
        train_accuracy = metrics.accuracy_score(true_classes, predicted_classes)
        with open(os.path.abspath(f'{results_directory}{parameters.experiment}_accuracies.txt'), 'a+') as output_accuracy_file:
            output_accuracy_file.write(f'{epoch}, {train_accuracy : 0.4f}, {val_accuracy : 0.4f} \n')

        # Check if we have decreased the validation error (in this case increased the Cohen's Kappa score)
        if val_loss <= parameters.best_val_error:
            # Save the state dictionaries of the network and the optimizer
            if parameters.parallel:
                torch.save(functions.convert_data_parallel(network), os.path.abspath(f'{network_directory}{parameters.experiment}.pth'))
            else:
                torch.save(network.state_dict(), os.path.abspath(f'{network_directory}{parameters.experiment}.pth'))
            torch.save(optimizer.state_dict(), os.path.abspath(f'{network_directory}{parameters.experiment}_optimizer.pth'))

            # Save the value that was the best validation error and save what epoch this value occurred at
            parameters.best_val_error = val_loss
            parameters.best_network_epoch = epoch

        # Save what the next epoch number would be (used for continuing training after stopping)
        parameters.epoch = epoch + 1

        # Save the parameter values to a file
        with open(os.path.abspath(f'{network_directory}{parameters.experiment}_parameters.pkl'), 'wb') as f:
            pickle.dump(parameters, f)

        if parameters.early_stopping:
            # If we have gone through several epochs without any improvement, commence early stopping
            if (epoch - parameters.best_network_epoch) > parameters.patience:
                print('Exceed patience, stopping early')
                return