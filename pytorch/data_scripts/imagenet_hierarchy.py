## Basic Python libraries
import sys
import os
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

    classes_synsets = [wordnet.synset_from_pos_and_offset(i[0], int(i[1:])).name() for i in classes]

    # Construct the correct hierarchy
    hierarchy = GeneralTree()

    # Loop through all terminal nodes and establish path from terminal to root
    for terminal_class in classes_synsets:
        terminal = hierarchy_info[terminal_class][0] 
        current = hierarchy_info[terminal_class][1] 

        # Check terminal (should always go into if)
        if not hasattr(hierarchy, terminal['name']):
            setattr(hierarchy, terminal['name'], TreeNode(terminal['wid'], terminal['name'], terminal['synset']))
        terminal_node = getattr(hierarchy, terminal['name']) 

        # Check parent of terminal
        if not hasattr(hierarchy, current['name']):
            setattr(hierarchy, current['name'], TreeNode(current['wid'], current['name'], current['synset']))
        current_node = getattr(hierarchy, current['name']) 

        terminal_node.set_parent(current_node) 
        
        # Create the ancestral path for this terminal node
        while current['name'] != 'entity.n.01':
            parent = hierarchy_info[current['name']][1] 
            if not hasattr(hierarchy, parent['name']):
                setattr(hierarchy, parent['name'], TreeNode(parent['wid'], parent['name'], parent['synset']))
            parent_node = getattr(hierarchy, parent['name']) 

            current_node.set_parent(parent_node)
            current = parent
            current_node = parent_node

    # Get root of the hierarchy
    root = getattr(hierarchy, 'entity.n.01') 

    # Ensure correctness in leaves
    leaves = root.leaves
    leaves = [leaf.name for leaf in leaves]
    terminals = [leaf in classes_synsets for leaf in leaves]
    print(f'Correct number of leaves?: {sum(terminals) == len(classes_synsets)}')

    # Ensure every leaf is connected to the root (entity)
    leaves = root.leaves
    entities = ['entity.n.01' in [ancestor.name for ancestor in leaf.ancestors] for leaf in leaves]
    print(f'All terminals have a path to entity?: {sum(entities) == len(classes_synsets)}')

    # Check for CIFAR10 classes
    cifar10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cifar10 = [wordnet.synsets(cifar10_class)[0].name() for cifar10_class in cifar10]
    all_nodes = [node.name for node in root.descendants]
    cifar10_classes_in_imagenet = [cifar10_class in all_nodes for cifar10_class in cifar10]

    # Check animal
    animal = getattr(hierarchy, wordnet.synsets('animal')[0].name())
    image_net_animal_classes = [terminal_animal.wid for terminal_animal in animal.leaves]
    for cl in image_net_animal_classes:
        print(f'\'{cl}\'', end=', ')
    print()

    for child in root.children:
        for grandchild in child.children:
            if 'object' in grandchild.name:
                for greatgrandchild in grandchild.children:
                    if 'whole' in greatgrandchild.name:
                        for greatgreatgrandchild in greatgrandchild.children:
                            DotExporter(greatgreatgrandchild).to_picture(f'{data_directory}imagenet_hierarchy_gggc_{greatgreatgrandchild.name}.png')
                            img = Image.open(f'{data_directory}imagenet_hierarchy_gggc_{greatgreatgrandchild.name}.png')
                            width = img.width
                            height = img.height
                            img = img.resize((width * 3, height * 3), Image.BICUBIC)
                            img.save(f'{data_directory}imagenet_hierarchy_gggc_{greatgreatgrandchild.name}.png')
                    else:
                        DotExporter(greatgrandchild).to_picture(f'{data_directory}imagenet_hierarchy_ggc_{greatgrandchild.name}.png')
                        img = Image.open(f'{data_directory}imagenet_hierarchy_ggc_{greatgrandchild.name}.png')
                        width = img.width
                        height = img.height
                        img = img.resize((width * 3, height * 3), Image.BICUBIC)
                        img.save(f'{data_directory}imagenet_hierarchy_ggc_{greatgrandchild.name}.png')
            else:
                DotExporter(grandchild).to_picture(f'{data_directory}imagenet_hierarchy_gc_{grandchild.name}.png')
                img = Image.open(f'{data_directory}imagenet_hierarchy_gc_{grandchild.name}.png')
                width = img.width
                height = img.height
                img = img.resize((width * 3, height * 3), Image.BICUBIC)
                img.save(f'{data_directory}imagenet_hierarchy_gc_{grandchild.name}.png')

    # Output tree into that nice ImageNet formatted txt file
    # TODO Later when needed

    # Check other subsets of ImageNet
    # furniture.n.01
    furniture = getattr(hierarchy, 'furniture.n.01')
    furniture_leaves = furniture.leaves
    furniture_num_classes = len(furniture_leaves)
    print(f'Num classes in ImageNet-Furniture: {furniture_num_classes}')

    # building.n.01
    building = getattr(hierarchy, 'building.n.01')
    building_leaves = building.leaves
    building_num_classes = len(building_leaves)
    print(f'Num classes in ImageNet-Building: {building_num_classes}')
    
    # self-propelled_vehicle.n.01
    vehicle = getattr(hierarchy, 'self-propelled_vehicle.n.01')
    vehicle_leaves = vehicle.leaves
    vehicle_num_classes = len(vehicle_leaves)
    print(f'Num classes in ImageNet-SPVehicle: {vehicle_num_classes}')
    
    # insect.n.01
    insect = getattr(hierarchy, 'insect.n.01')
    insect_leaves = insect.leaves
    insect_num_classes = len(insect_leaves)
    print(f'Num classes in ImageNet-Insect: {insect_num_classes}')
    
    # fish.n.01
    fish = getattr(hierarchy, 'fish.n.01')
    fish_leaves = fish.leaves
    fish_num_classes = len(fish_leaves)
    print(f'Num classes in ImageNet-Fish: {fish_num_classes}')
    
    # carnivore.n.01
    carnivore = getattr(hierarchy, 'carnivore.n.01')
    carnivore_leaves = carnivore.leaves
    carnivore_num_classes = len(carnivore_leaves)
    print(f'Num classes in ImageNet-Carnivore: {carnivore_num_classes}')
    
    # placental.n.01
    placental = getattr(hierarchy, 'placental.n.01')
    placental_leaves = placental.leaves
    placental_num_classes = len(placental_leaves)
    print(f'Num classes in ImageNet-Placental: {placental_num_classes}')
    
    # dog.n.01
    dog = getattr(hierarchy, 'dog.n.01')
    dog_leaves = dog.leaves
    dog_num_classes = len(dog_leaves)
    print(f'Num classes in ImageNet-Dog: {dog_num_classes}')
    
    # matter.n.03 (food.n.01 & food.n.02)
    matter = getattr(hierarchy, 'matter.n.03')
    matter_leaves = matter.leaves
    matter_num_classes = len(matter_leaves)
    print(f'Num classes in ImageNet-Matter: {matter_num_classes}')
    
    # geological_formation.n.01
    geological_formation = getattr(hierarchy, 'geological_formation.n.01')
    geological_formation_leaves = geological_formation.leaves
    geological_formation_num_classes = len(geological_formation_leaves)
    print(f'Num classes in ImageNet-Geological_Formation: {geological_formation_num_classes}')

    print()

    ## After analysis of the above subsets, look into ImageNet-SPVehicles
    instances_per_class = {}
    for terminal in vehicle_leaves:
        terminal_wid = terminal.wid
        class_tar = tarfile.open(f'{image_directory}train/{terminal_wid}/{terminal_wid}.tar')
        train_instances = class_tar.getnames()
        num_instances = len(train_instances)
        instances_per_class[terminal.name] = [terminal_wid, num_instances]
    
    print('ImageNet-SPVehicle:')
    for class_name, num_instances in instances_per_class.items():
        print(f'\t{class_name : <30} : {num_instances}')

    print('SPV Classes:', end=' ')
    for class_name, num_instances in instances_per_class.items():
        print(f'\'{instances_per_class[class_name][0]}\'', end=', ')

    print()

    ## After analysis of the above subsets, look into ImageNet-Furniture
    instances_per_class = {}
    for terminal in furniture_leaves:
        terminal_wid = terminal.wid
        class_tar = tarfile.open(f'{image_directory}train/{terminal_wid}/{terminal_wid}.tar')
        train_instances = class_tar.getnames()
        num_instances = len(train_instances)
        instances_per_class[terminal.name] = [terminal_wid, num_instances]
    
    print('ImageNet-Furniture:')
    for class_name, num_instances in instances_per_class.items():
        print(f'\t{class_name : <30} : {num_instances}')
    print('Furniture Classes:', end=' ')
    for class_name, num_instances in instances_per_class.items():
        print(f'\'{instances_per_class[class_name][0]}\'', end=', ')

    print()

    ## After analysis of the above subsets, look into ImageNet-Matter
    instances_per_class = {}
    for terminal in matter_leaves:
        terminal_wid = terminal.wid
        class_tar = tarfile.open(f'{image_directory}train/{terminal_wid}/{terminal_wid}.tar')
        train_instances = class_tar.getnames()
        num_instances = len(train_instances)
        instances_per_class[terminal.name] = [terminal_wid, num_instances]
    
    print('ImageNet-Matter:')
    for class_name, num_instances in instances_per_class.items():
        print(f'\t{class_name : <30} : {num_instances}')

    print()

    ## After analysis of the above subsets, look into ImageNet-Buildings
    instances_per_class = {}
    for terminal in building_leaves:
        terminal_wid = terminal.wid
        class_tar = tarfile.open(f'{image_directory}train/{terminal_wid}/{terminal_wid}.tar')
        train_instances = class_tar.getnames()
        num_instances = len(train_instances)
        instances_per_class[terminal.name] = [terminal_wid, num_instances]
    
    print('ImageNet-Building:')
    for class_name, num_instances in instances_per_class.items():
        print(f'\t{class_name : <30} : {num_instances}')

    print('done')
