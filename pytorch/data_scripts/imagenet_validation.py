## Basic Python libraries
import sys
import os
import shutil
import subprocess
import tarfile
import yaml
import pickle
sys.path.append(os.getcwd() + '/')

## Deep learning and array processing libraries
import numpy as np 
import pandas as pd 

## Tree-Related Imports
from anytree import NodeMixin
from anytree import RenderTree
from anytree.exporter import DotExporter

## Other imports
from PIL import Image
import scipy
from scipy.io import loadmat
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

class GeneralTree(object):
    def __init__(self):
        pass
    
    def __repr__(self):
        root = getattr(self, 'entity.n.01')
        return str(RenderTree(root))

    def output_txt(self):
        root = getattr(self, 'entity.n.01')
        print('pause')

class TreeNode(NodeMixin):
    def __init__(self, wid, name, synset, parent=None):
        super(TreeNode, self).__init__()
        self.wid = wid
        self.name = name
        self.synset = synset
        self.parent = parent

    def set_parent(self, parent):
        if isinstance(parent, TreeNode):
            self.parent = parent
        else:
            raise Exception('parent must be of type TreeNode')

    def __eq__(self, other):
        if isinstance(other, TreeNode):
            return self.wid == other.wid
        else:
            raise Exception('other must be of type TreeNode')

    def __repr__(self):
        return f'[{self.wid}: {self.name}: {self.synset}]'

if __name__ == '__main__':
    dataset = 'imagenet'

    if sys.platform == 'win32':
        config_path = f'./config/logan_pc_{dataset}.yaml'
    elif sys.platform == 'darwin':
        config_path = f'./config/logan_mac_{dataset}.yaml'
    elif sys.platform == 'linux':
        config_path = f'./config/logan_linux_{dataset}.yaml'

    dataset_image_directory = f'{dataset}_image_directory'
    dataset_network_directory = f'{dataset}_network_directory'
    dataset_data_directory = f'{dataset}_data_directory'
    dataset_results_directory = f'{dataset}_results_directory'

    # Open the yaml config file
    try:
        with open(os.path.abspath(config_path)) as config_file: 
            config = yaml.safe_load(config_file)

            # Location of root directory of all images
            image_directory = config['Paths'][dataset_image_directory]

            # Location of network parameters (network state dicts, etc.)
            network_directory = config['Paths'][dataset_network_directory]

            # Location of parsed data (dataframes, etc.)
            data_directory = config['Paths'][dataset_data_directory]

            # Location of saved results from evaluation (confusion matrix, etc.)
            results_directory = config['Paths'][dataset_results_directory]

    except:
        raise Exception('Error loading data from config file.')

    # Load the IS-A relationships
    relationships = np.loadtxt(f'{data_directory}wordnet.is_a.txt', dtype=str)

    hierarchy_info = {}

    for relationship in relationships:
        parent_wid = relationship[0]
        parent_pos = parent_wid[0]
        parent_offset = int(parent_wid[1:])
        parent_synset = wordnet.synset_from_pos_and_offset(parent_pos, parent_offset)
        parent_name = parent_synset.name()

        child_wid = relationship[1]
        child_pos = child_wid[0]
        child_offset = int(child_wid[1:])
        child_synset = wordnet.synset_from_pos_and_offset(child_pos, child_offset)
        child_name = child_synset.name()

        parent = {'wid': parent_wid, 'name': parent_name, 'synset': parent_synset}
        child = {'wid': child_wid, 'name': child_name, 'synset': child_synset}

        hierarchy_info[child['name']] = [child, parent]

    # Load the WordNet IDs of all terminal classes
    classes = os.listdir(f'{image_directory}train/')
    classes = [cl.strip('.tar') for cl in classes]

    # Get validation data
    instance_map = np.loadtxt(f'{data_directory}devkit/data/ILSVRC2012_validation_ground_truth.txt', dtype=str)
    index_to_wid_map = np.loadtxt(f'{data_directory}id_to_class_map.txt', dtype=str)[:, :2]
    index_to_wid_map = {index : wid for (wid, index) in index_to_wid_map}
    validation_instances = [instance for instance in os.listdir(f'{image_directory}val/') if instance[-5:] == '.JPEG']

    # Start mapping and moving
    for index, (class_index, instance_name) in enumerate(zip(instance_map, validation_instances)):
        instance_wid = index_to_wid_map[class_index]
        initial_location = f'{image_directory}val/{instance_name}'
        new_location = f'{image_directory}val/{instance_wid}/{instance_name}'
        shutil.move(initial_location, new_location)

    print('done')
