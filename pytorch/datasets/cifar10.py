## PyTorch imports
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class CIFAR10(Dataset):
    """
    A different class for the CIFAR10 dataset, allows me to control it a little more than the
    base TorchVision one.

    :param normalization_values: the different mean and std-devs for each of the different 
                                 pixel dropout kill rates
    :param classes_index: the integer class index for each instance in the dataset
    :param classes: the unique integers in classes_index
    :param classes_names: the unique class names in string format
    :param classes_to_index: maps string class name to integer index
    :param instances: the image data for each instance
    :param phase: what phase (train, val, or test)
    :param transform: what transform / data augmentation to apply to each instance when laoded
    :param visualize: a boolean flag for whether or not we should display each loaded instance
    """

    # The different mean and std-devs for each of the different pixel dropout kill rates
    normalization_values = {
        -1.0 : {'mean' : [0.4913, 0.4823, 0.4465], 'std' : [0.2024, 0.1994, 0.2010]}
    }

    def __init__(self, classes_index, classes_names, classes_to_index, instances, phase, transform=None, visualize=False):
        self.classes_index = classes_index
        self.classes = torch.unique(self.classes_index)
        self.classes_names = classes_names
        self.classes_to_index = classes_to_index
        self.instances = instances
        self.phase = phase
        self.transform = transform
        self.visualize = visualize

    def num_classes(self):
        """
        Returns the number of classes in the dataset
        """
        return len(self.classes)

    def __len__(self):
        """
        Returns the number of instances in the dataset object
        """
        return len(self.instances)
    
    def __getitem__(self, index):
        """
        Loads an individual instance and its label.

        :return: image data
        :return: ground truth label corresponding to the image
        """
        image = self.instances[index]
        image = Image.fromarray(image)
        label = self.classes_index[index].numpy().astype(np.int64)

        if self.transform:
            image = self.transform(image)

        if self.visualize:
            image = np.asarray(image)
            fig = plt.figure()
            plt.imshow(image)
            plt.title(self.classes_names[label])
            plt.show()

        return (image, label)

    @staticmethod
    def create_validation(train, test, visualize=False):
        """
        Creates a validation set since only train and test are available for CIFAR10. Creates the
        validation set by sampling from the training set (and removing)

        :param train:
        :param test:
        :param visualize:
        """

        # For some reason train.targets and test.targets are torch tensors in MNIST, but python lists for CIFAR10
        train.targets = torch.tensor(train.targets)
        test.targets = torch.tensor(test.targets)

        ordered_labels, indexes = torch.sort(train.targets)
        ordered_data = train.data[indexes]

        uniques, counts = torch.unique(test.targets, return_counts=True)

        train_targets = torch.empty(0, dtype=torch.int64)
        train_instances = np.empty([0, 32, 32, 3], dtype=np.uint8)
        val_targets = torch.empty(0, dtype=torch.int64)
        val_instances = np.empty([0, 32, 32, 3], dtype=np.uint8)

        for num in uniques:
            num = num.item()
            data = ordered_data[ordered_labels == num]
            labels = ordered_labels[ordered_labels == num]

            train_data, val_data = np.split(data, [len(data) - counts[num]])
            train_labels, val_labels = torch.split(labels, [len(data) - counts[num], counts[num]])

            train_targets = torch.cat([train_targets, train_labels])
            train_instances = np.concatenate([train_instances, train_data])
            val_targets = torch.cat([val_targets, val_labels])
            val_instances = np.concatenate([val_instances, val_data])

        del train_data, train_labels, val_data, val_labels

        train_dataset = CIFAR10(train_targets, train.classes, train.class_to_idx, train_instances, phase='train', transform=train.transform, visualize=visualize)
        val_dataset = CIFAR10(val_targets, train.classes, train.class_to_idx, val_instances, phase='val', transform=test.transform, visualize=visualize)

        return train_dataset, val_dataset
    
    @staticmethod
    def create_train(train, visualize=False):
        test_dataset = CIFAR10(train.targets, train.classes, train.class_to_idx, train.data, phase='train', transform=train.transform, visualize=visualize)
        return test_dataset

    @staticmethod
    def create_test(test, phase='test', visualize=False):
        """
        Takes the TorchVision CIFAR10 class objects and converts to my CIFAR10 Dataset object
        """
        test_dataset = CIFAR10(test.targets, test.classes, test.class_to_idx, test.data, phase='test', transform=test.transform, visualize=visualize)
        return test_dataset