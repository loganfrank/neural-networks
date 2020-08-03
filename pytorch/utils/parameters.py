## Basic Python imports
import math

class Parameters():
    """
    This class is used as a data structure for storing parameters such as experiment name, hyperparameters, etc.
    """

    def __init__(self, experiment='baseline', script=None, **experiment_args):
        # Name of the experiment, affects the names files are stored as
        self.experiment = experiment

        # What script are we running
        self.script = script

        # Add our command line arguments
        for arg, value in experiment_args.items():
            setattr(self, arg, value)

        # The position to start training at, useful if need to stop training and continue later
        self.epoch = 0

        # The epoch at which we saw the best performance from the network
        self.best_network_epoch = 0

        # The best metric scores seen on the validation data (currently using Cohen's Kappa)
        self.best_val_error = math.inf

    def __str__(self):
        parameters = vars(self)
        return '\n'.join([f'{arg} = {value}' for arg, value in parameters.items()])
                
