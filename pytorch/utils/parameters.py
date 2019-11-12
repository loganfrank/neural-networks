## Basic Python imports
import math

class Parameters():
    """
    This class is used as a data structure for storing parameters such as experiment name, hyperparameters, etc.
    """

    def __init__(self, experiment='baseline', epoch=0, num_epochs=1000, batch_size=8, learning_rate=1.0e-3, weight_decay=1.0e-5, momentum=0.9, best_network_epoch=0, patience=10, best_val_error=math.inf, early_stopping=False, parallel=False):
        # Name of the experiment, affects the names files are stored as
        self.experiment = experiment

        # The position to start training at, useful if need to stop training and continue later
        self.epoch = epoch

        # The maximum number of epochs we will train for (regardless of early stopping)
        self.num_epochs = num_epochs

        # What batch size we used for training
        self.batch_size = batch_size

        # The initial learning rate used for training
        self.learning_rate = learning_rate

        # Weight decay value, prevents weights from reaching extreme values
        self.weight_decay = weight_decay

        # Momentum
        self.momentum = momentum

        # The epoch at which we saw the best performance from the network
        self.best_network_epoch = best_network_epoch

        # The number of epochs that will pass before early stopping engages (i.e. the number of epochs allowed where no improvement is made)
        self.patience = patience

        # The best metric scores seen on the validation data (currently using Cohen's Kappa)
        self.best_val_error = best_val_error

        # Flag for whether or not to use early stopping in training
        self.early_stopping = early_stopping

        # Are we using a DataParallel model
        self.parallel = parallel

    def __str__(self):
        return (f'Experiment: {self.experiment}\n' + 
                f'Number of Epochs: {self.num_epochs}\n' +
                f'Batch Size: {self.batch_size}\n' +
                f'Learning Rate: {self.learning_rate}\n' +
                f'Weight Decay: {self.weight_decay}\n' +
                f'Patience: {self.patience}\n' +
                f'Early Stopping: {self.early_stopping}\n' +
                f'Data Parallel: {self.parallel} \n')
                
