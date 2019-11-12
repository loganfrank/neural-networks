## Basic Python imports
import os

## PyTorch imports
from torch.utils.data import Dataset
from torchvision import transforms

## Image / Array imports
import numpy as np
from PIL import Image

class cut_dataset(Dataset):
    
    def __init__(self, image_root_directory, dataframe, transform=None, phase='train', balance=False, cut=None):
        # The root directory containing all image data for a desired set/phase
        self.image_root_directory = os.path.abspath(f'{image_root_directory}{phase}/')
        dataframe_subset = dataframe.loc[phase]

        # Get the list of instance file names
        self.instances = dataframe_subset.index.get_level_values(1).to_numpy().astype('U')

        # Get the crop associated with each instance
        self.crops = dataframe_subset.index.get_level_values(0).to_numpy().astype('U')

        # Get the ground truth class labels for each instance, define the set of unique output classes and assign them an integer index
        self.classes = dataframe_subset.to_numpy().astype('U')
        self.classes = np.squeeze(self.classes)
        self.classes_unique, classes_counts = np.unique(self.classes, return_counts=True)
        self.classes_to_index = {}
        for index, value in enumerate(self.classes_unique):
            self.classes_to_index[value] = index

        # Balance the dataset if flagged to do so
        if balance:

            # Get the number of instances in the class with the largest number of instances
            max_class = max(classes_counts)

            # Reset our class variables for the lists of instance image names, instance class labels, and instance crop labels
            self.instances = np.empty(0)
            self.classes = np.empty(0)
            self.crops = np.empty(0)

            # Create 2D matrices where each row is a class and each column is an instance of that class
            if cut is None:
                instances_2D = np.empty((0, max_class))
                labels_2D = np.empty((0, max_class))
                crops_2D = np.empty((0, max_class))
            elif isinstance(cut, int) and cut > 0:
                instances_2D = np.empty((0, cut))
                labels_2D = np.empty((0, cut))
                crops_2D = np.empty((0, cut))
            else:
                raise Exception('Invalid value for cut - must be either None or an int greater than 0.')

            # Loop through each possible class and perform the balancing computations
            for row_index, class_value in enumerate(self.classes_unique):
                # Get the instance image names, instance class labels, and instance crop labels for each class
                class_instances = dataframe_subset[dataframe_subset['disease'] == class_value].index.get_level_values(1).to_numpy().astype('U')
                class_labels = dataframe_subset[dataframe_subset['disease'] == class_value].to_numpy().astype('U')
                class_crops = dataframe_subset[dataframe_subset['disease'] == class_value].index.get_level_values(0).to_numpy().astype('U')
                
                # Create the array used for doing the same shuffle on all arrays
                shuffle = np.arange(class_instances.shape[0])
                np.random.shuffle(shuffle)

                # Shuffle the lists of instance image names, instance class labels, and instance crop labels
                class_instances = class_instances[shuffle]
                class_labels = class_labels[shuffle]
                class_crops = class_crops[shuffle]
                
                # Concatenate the arrays until the exceed the largest number of instances in a class
                while len(class_instances) < max_class:
                    class_instances = np.concatenate((class_instances, class_instances))
                    class_labels = np.concatenate((class_labels, class_labels))
                    class_crops = np.concatenate((class_crops, class_crops))

                # Crop the end so the class now has a number of instances equal to the largest number of instances in a class
                if cut is None:
                    class_instances = class_instances[:max_class]
                    class_labels = class_labels[:max_class]
                    class_crops = class_crops[:max_class]
                else:
                    class_instances = class_instances[:cut]
                    class_labels = class_labels[:cut]
                    class_crops = class_crops[:cut]

                # Concatenate / append to the rows of the 2D matrices
                instances_2D = np.concatenate((instances_2D, class_instances.reshape(1, -1)), axis=0)
                labels_2D = np.concatenate((labels_2D, class_labels.reshape(1, -1)), axis=0)
                crops_2D = np.concatenate((crops_2D, class_crops.reshape(1, -1)), axis=0)

            # Segment data into batches
            # Below is the number of instances per class in each batch (in this class batch size 30)
            num_per_class = 3 
            for i in range(0, (len(class_instances) // num_per_class) * num_per_class, num_per_class):
                # Extract one batch worth of information
                batch_instances = instances_2D[:, i : i + num_per_class]
                batch_labels = labels_2D[:, i : i + num_per_class]
                batch_crops = crops_2D[:, i : i + num_per_class]

                # Add to global list
                self.instances = np.concatenate((self.instances, batch_instances.flatten().reshape(-1, 1)), axis=None) # TODO is this right
                self.classes = np.concatenate((self.classes, batch_labels.flatten().reshape(-1, 1)), axis=None)
                self.crops = np.concatenate((self.crops, batch_crops.flatten().reshape(-1, 1)), axis=None)
                
            # If the length of any of the above global arrays is not equal to the number of instances in the largest
            # class times the number of classes, then more data remains to append to the global list
            if len(self.instances) < (len(class_instances) * len(self.classes_unique)):
                i += num_per_class
                batch_instances = instances_2D[:, i:]
                batch_labels = labels_2D[:, i:]
                batch_crops = crops_2D[:, i:]
                self.instances = np.concatenate((self.instances, batch_instances.flatten().reshape(-1, 1)), axis=None)
                self.classes = np.concatenate((self.classes, batch_labels.flatten().reshape(-1, 1)), axis=None)
                self.crops = np.concatenate((self.crops, batch_crops.flatten().reshape(-1, 1)), axis=None) 

        # Map each instance's ground truth class label to its associated integer index
        self.classes_index = np.zeros((len(self.instances)))
        for index, class_name in enumerate(self.classes):
            self.classes_index[index] = self.classes_to_index[class_name]

        # Save the transform to apply to the data before loading
        self.transform = transform

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        # Open the input image using PIL
        image = Image.open(os.path.abspath(f'{self.image_root_directory}/{self.crops[index]}/{self.classes[index]}/{self.instances[index]}'))

        # Identify the class label and convert it to long
        class_index = self.classes_index[index]
        class_index = class_index.astype(np.int64)

        # Perform the data transform on the image
        if self.transform:
            image = self.transform(image)

        return (image, class_index)


class dataset(Dataset):
    
    def __init__(self, image_root_directory, dataframe, transform=None, phase='train'):
        # The root directory containing all image data for a desired set/phase
        self.image_root_directory = os.path.abspath(f'{image_root_directory}{phase}/')
        dataframe_subset = dataframe.loc[phase]

        # Get the list of instance file names
        self.instances = dataframe_subset.index.get_level_values(1).to_numpy().astype('U')

        # Get the crop associated with each instance
        self.crops = dataframe_subset.index.get_level_values(0).to_numpy().astype('U')

        # Get the ground truth class labels for each instance, define the set of unique output classes and assign them an integer index
        self.classes = dataframe_subset.to_numpy().astype('U')
        self.classes = np.squeeze(self.classes)
        self.classes_unique = np.unique(self.classes)
        self.classes_to_index = {}
        for index, value in enumerate(self.classes_unique):
            self.classes_to_index[value] = index

        # Map each instance's ground truth class label to its associated integer index
        self.classes_index = np.zeros((len(self.instances)))
        for index, class_name in enumerate(self.classes):
            self.classes_index[index] = self.classes_to_index[class_name]

        # Save the transform to apply to the data before loading
        self.transform = transform

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        # Open the input image using PIL
        image = Image.open(os.path.abspath(f'{self.image_root_directory}/{self.crops[index]}/{self.classes[index]}/{self.instances[index]}'))

        # Identify the class label and convert it to long
        class_index = self.classes_index[index]
        class_index = class_index.astype(np.int64)

        # Perform the data transform on the image
        if self.transform:
            image = self.transform(image)

        return (image, class_index)
