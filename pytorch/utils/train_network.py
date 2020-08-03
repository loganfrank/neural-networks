## Basic Python imports
import sys
import os

## Deep learning and array processing imports
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn

## Inter-project imports
from utils import evaluate_network
from utils.recorder import TrainingRecorder

## Metrics imports
import sklearn.metrics as metrics

def train_network(network, optimizer, scheduler, parameters, train_dataloader, val_dataloader, compute_device, network_directory, results_directory):
    # Create classification loss function
    classification_loss_func = nn.CrossEntropyLoss()

    # Create a recorder
    recorder = TrainingRecorder(results_directory, network_directory, parameters, new=(parameters.epoch == 0))

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
            elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(running_loss)

        # Get the training accuracy
        loss = running_loss / (batch_num + 1)
        accuracy = metrics.accuracy_score(true_classes, predicted_classes)

        # Check the validation error after each training epoch
        print(f'Evaluating validation set (epoch {epoch}):')
        val_loss, val_accuracy = evaluate_network.validate_network(network=network, dataloader=val_dataloader, compute_device=compute_device, 
                                                            experiment=parameters.experiment, results_directory=results_directory, classification_loss_func=classification_loss_func)

        recorder.record(epoch, loss, val_loss, accuracy, val_accuracy)
        recorder.update(epoch, val_loss, network.state_dict(), optimizer.state_dict())
