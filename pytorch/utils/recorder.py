import torch
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sn
import sklearn.metrics as metrics

class TrainingRecorder:
    def __init__(self, results_directory, network_directory, parameters, new=True):
        self.results_directory = results_directory
        self.network_directory = network_directory
        self.parameters = parameters
        self.experiment = parameters.experiment
        if new:
            self._start()

    def _start(self):  
        # Make the directory if it doesn't exists 
        if not os.path.exists(os.path.abspath(f'{self.results_directory}{self.experiment}')): 
            os.makedirs(os.path.abspath(f'{self.results_directory}{self.experiment}'))

        # Make the files
        with open(os.path.abspath(f'{self.results_directory}{self.experiment}/{self.experiment}_average_losses.txt'), 'w') as output_running_loss_file:
            output_running_loss_file.write(f'{self.experiment}\n')
            output_running_loss_file.write('(epoch number), (average training loss for given epoch number), (average validation loss for given epoch number) \n')

        with open(os.path.abspath(f'{self.results_directory}{self.experiment}/{self.experiment}_accuracies.txt'), 'w') as output_accuracy_file:
            output_accuracy_file.write(f'{self.experiment}\n')
            output_accuracy_file.write('(epoch number), (training accuracy for given epoch number), (validation accuracy for given epoch number) \n')

    def record(self, epoch, loss, val_loss, accuracy, val_accuracy):
        # Append the epoch number and the average loss for that epoch to the previously created average loss output file
        with open(os.path.abspath(f'{self.results_directory}{self.experiment}/{self.experiment}_average_losses.txt'), 'a+') as output_running_loss_file:
            output_running_loss_file.write(f'{epoch}, {loss : 0.4f}, {val_loss : 0.4f} \n')
        
        with open(os.path.abspath(f'{self.results_directory}{self.experiment}/{self.experiment}_accuracies.txt'), 'a+') as output_accuracy_file:
            output_accuracy_file.write(f'{epoch}, {accuracy : 0.4f}, {val_accuracy : 0.4f} \n')

    def update(self, epoch, val_loss, network_dict, optimizer_dict):
        # Check if we have decreased the validation error
        if val_loss <= self.parameters.best_val_error:
            # Save the state dictionaries of the network and the optimizer
            torch.save(network_dict, os.path.abspath(f'{self.network_directory}{self.experiment}/{self.experiment}.pth'))
            torch.save(optimizer_dict, os.path.abspath(f'{self.network_directory}{self.experiment}/{self.experiment}_optimizer.pth'))

            # Save the value that was the best validation error and save what epoch this value occurred at
            self.parameters.best_val_error = val_loss
            self.parameters.best_network_epoch = epoch

        # Save what the next epoch number would be (used for continuing training after stopping)
        self.parameters.epoch = epoch + 1

        # Save the network weights after each epoch
        torch.save(network_dict, os.path.abspath(f'{self.network_directory}{self.experiment}/{self.experiment}_weights_epoch{epoch}.pth'))

        # Save the parameter values to a file
        with open(os.path.abspath(f'{self.network_directory}{self.experiment}/{self.experiment}_parameters.pkl'), 'wb') as f:
            pickle.dump(self.parameters, f)


class EvaluateRecorder:
    def __init__(self, results_directory, experiment, phase):
        self.results_directory = results_directory
        self.experiment = experiment
        self.phase = phase

    def record(self, true_classes, predicted_classes, classes, save=False):
        # Compute the general accuracy, F1 score, Cohen's Kappa score, and confusion matrix for the data and print each
        accuracy = metrics.accuracy_score(true_classes, predicted_classes)
        weighted_accuracy = metrics.balanced_accuracy_score(true_classes, predicted_classes)
        precision = metrics.precision_score(true_classes, predicted_classes, average='micro')
        recall = metrics.recall_score(true_classes, predicted_classes, average='micro')
        cohens_kappa = metrics.cohen_kappa_score(true_classes, predicted_classes)
        confusion_matrix = metrics.confusion_matrix(true_classes, predicted_classes)
        confusion_matrix = (confusion_matrix.T / np.sum(confusion_matrix, axis=1)).T
        
        print(f'Accuracy: {accuracy : 0.4f}')
        print(f'Weighted Accuracy: {weighted_accuracy : 0.4f}')
        print('Accuracy by class:')
        for class_name, class_accuracy in zip(classes, confusion_matrix.diagonal()):
            print(f'\tClass {class_name}: { class_accuracy : 0.4f}')
        print(f'Precision: {precision : 0.4f}') 
        print(f'Recall: {recall : 0.4f}') 
        print(f'Cohen\'s Kappa: {cohens_kappa : 0.4f}')
        print(f'Confusion Matrix: \n{np.round(confusion_matrix, decimals=3)}')

        # Save the resulting metrics to a file (if desired)
        if save:
            # Make the directory if it doesn't exist
            if not os.path.exists(os.path.abspath(f'{self.results_directory}{self.experiment}/{self.experiment}')): 
                os.makedirs(os.path.abspath(f'{self.results_directory}{self.experiment}/{self.experiment}'))

            # Save the general accuracy, F1 score, and Cohen's Kappa score
            with open(os.path.abspath(f'{self.results_directory}{self.experiment}/{self.experiment}/{self.experiment}_results_{self.phase}.txt'), 'w') as f:
                f.write(f'Accuracy: {accuracy : 0.4f} \n')
                f.write(f'Weighted Accuracy: {weighted_accuracy : 0.4f} \n')
                f.write('Accuracy by class: \n')
                for class_name, class_accuracy in zip(classes, confusion_matrix.diagonal()):
                    f.write(f'\tClass {class_name}: { class_accuracy : 0.4f} \n')
                f.write(f'Precision: {precision : 0.4f} \n')
                f.write(f'Recall: {recall : 0.4f} \n')
                f.write(f'Cohen\'s Kappa: {cohens_kappa : 0.4f} \n')
                f.write('---\n')
                f.write(f'Confusion Matrix: \n{np.round(confusion_matrix, decimals=3)} \n')

            # Create and save the confusion matrix
            cm_df = pd.DataFrame(confusion_matrix, index=classes, columns=classes)
            fig, ax = plt.subplots(1, 1, figsize=(20, 15))
            sn.heatmap(cm_df, cmap='YlGnBu', cbar=False, annot=True, ax=ax)
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
            ax.set_title('Val Set Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            fig.savefig(os.path.abspath(f'{self.results_directory}{self.experiment}/{self.experiment}/{self.experiment}_confusion_matrix_{self.phase}.png'))

        if self.phase == 'val':
            return accuracy
        else:
            return accuracy, weighted_accuracy, precision, recall, cohens_kappa, confusion_matrix