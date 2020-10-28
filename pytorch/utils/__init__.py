import sys

import torch
import torch.nn.functional as F

###############################################
##### Bold Driver Learning Rate Scheduler #####
###############################################

class BoldDriver():
    def __init__(self, optimizer, network, decrease_factor=0.5, increase_factor=1.2):
        """ Decrease factor should be < 1 and increase factor should be > 1 """
        self.optimizer = optimizer
        self.network = network
        self.last_optim_state = optimizer.state_dict()
        self.last_network_state = network.state_dict()
        self.decrease_factor = decrease_factor
        self.increase_factor = increase_factor
        self.num_fails = 0
        # set last loss to max, so it must improve
        self.lastLoss = sys.float_info.max

    def step(self, loss):
        # increase lr if we are improving, decrease it if we are not
        if loss < self.lastLoss:
            self._update_lr(self.increase_factor)
            # Save the state
            self.last_optim_state = self.optimizer.state_dict()
            self.last_network_state = self.network.state_dict()
            self.num_fails = 0
            self.lastLoss = loss
        else:
            # Revert to last state, then decrease the learning rate
            self.optimizer.load_state_dict(self.last_optim_state)
            self.network.load_state_dict(self.last_network_state)
            self._update_lr(self.decrease_factor)
            self.num_fails += 1
        return self.num_fails

    def state_dict(self):
        return {'lastLoss': self.lastLoss, 'decrease_factor':
                self.decrease_factor, 'increase_factor': self.increase_factor,
                'num_fails': self.num_fails}

    def load_state_dict(self, dict):
        self.lastLoss = dict['lastLoss']
        self.decrease_factor = dict['decrease_factor']
        self.increase_factor = dict['increase_factor']
        self.num_fails = dict['num_fails']

    def _update_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * factor

##########################################
##### Class Activation Map Generator #####
##########################################

def class_activation_map(batch_feature_maps, weights, image_size=None):
    """
    From learning deep features for discriminative localization: http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf

    :param batch_feature_maps: the matrix containing all feature maps for every example in the batch (e.g. ResNet18 -> [batch_size, 512, h, w])
    :param weights: the weights in the FC layer of the network
    :param image_size: the original size of the image before going into the network, used for interpolating if desired (None is not interpolation)
    """
    
    # Get the dimensions of the feature maps and allocate our output matrix
    batch_size, num_channels, height, width = batch_feature_maps.shape
    output_class_activation_maps = torch.empty(batch_size, weights.shape[0], height, width)

    # Iterate through every instance in the batch (more than likely this will be a batch of 1 but I wanted to make it generic)
    for batch_index, feature_maps in enumerate(batch_feature_maps):

        # Rasterize each feature map 
        feature_maps = feature_maps.reshape((num_channels, height * width))

        # Iterate over every possible class
        for class_index in range(weights.shape[0]):

            # Matrix multiply the FC weights and feature maps
            instance_class_activation_map = weights[class_index] @ feature_maps

            # Reshape from rasterized vector back to 2D image
            instance_class_activation_map = instance_class_activation_map.reshape((height, width))

            # Normalize between 0 and 1
            instance_class_activation_map = instance_class_activation_map - instance_class_activation_map.min()
            instance_class_activation_map = instance_class_activation_map / instance_class_activation_map.max()

            # Add to output matrix of class_activation_maps
            output_class_activation_maps[batch_index, class_index, :, :] = instance_class_activation_map

    # If an image size is provided, resize to match the original image size
    if image_size is not None:
        output_class_activation_maps = F.interpolate(output_class_activation_maps, size=image_size, mode='bilinear', align_corners=True)

    return output_class_activation_maps

##############################################################################
##### Helper Function for Running Through Validation Set During Training #####
##############################################################################

def validation(network, dataloader, compute_device, experiment, results_directory, classification_loss_func, save=False, get_features=False):

    # Get the batch size
    batch_size = dataloader.batch_sampler.batch_size

    # Instantiate two arrays to keep track of the ground truth label and network prediction for each instance
    num_instances = len(dataloader.dataset)
    true_classes = np.zeros(num_instances)
    predicted_classes = np.zeros(num_instances)

    if get_features:
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
            predictions, _, _ = network(image_data, 'clean')

            # Accumulate the validation loss for each batch
            loss = classification_loss_func(predictions, label_data)
            val_loss += loss.item()

            # Get the flat prediction
            predictions = torch.argmax(predictions, dim=1)

            # Record the actual and predicted labels for the instance
            true_classes[ batch_num * batch_size : min( (batch_num + 1) * batch_size, num_instances) ] = label_data.detach().cpu().numpy()
            predicted_classes[ batch_num * batch_size : min( (batch_num + 1) * batch_size, num_instances) ] = predictions.detach().cpu().numpy() 

            if get_features:
                features.append(instance_features.squeeze().detach().cpu().numpy())

    recorder = EvaluateRecorder(results_directory, experiment, 'val')
    accuracy = recorder.record(true_classes, predicted_classes, dataloader.dataset.classes, save=save)
    
    return (val_loss / (batch_num + 1)), accuracy