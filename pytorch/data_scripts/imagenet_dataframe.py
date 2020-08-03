## Basic Python libraries
import sys
import os
import argparse
import yaml
sys.path.append(os.getcwd() + '/')

## Deep learning and array processing libraries
import numpy as np 
import pandas as pd 

## Other imports
import tarfile
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

# Inner project imports
from datasets import imagenet

def extract_tar(path, wid):
    tar_file_path = f'{path}{wid}.tar'
    tar_file = tarfile.open(tar_file_path)
    tar_file.extractall(path)
    tar_file.close()  
    del tar_file_path, tar_file 

def create_dataframe(path, output_path, dataset, wids):

    # Initialize DataFrame with its index and column names
    index = pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=['phase', 'wid', 'instance'])
    dataframe = pd.DataFrame(index=index, columns=['synset'])

    # Loop through each set of data (i.e. train, val)
    phases = os.listdir(path)
    for phase in phases:
        phase_path = f'{path}{phase}/'

        # Skip over files (we only want directories at this point)
        if not os.path.isdir(phase_path):
            print(f'{crop_class_path} is not a directory!')
            continue

        # Loop over desired classes
        for num, wid in enumerate(wids):
            wid_path = f'{phase_path}{wid}/'

            # Skip over files (we only want directories at this point)
            if not os.path.isdir(wid_path):
                print(f'{wid_path} is not a directory!')
                continue

            synset = wordnet.synset_from_pos_and_offset(wid[0], int(wid[1:])).name()

            if phase == 'train' and len(os.listdir(wid_path)) < 5:
                # Extract contents from the tar file
                extract_tar(wid_path, wid)

            # Loop through each instance in the disease, add the instance as a row in the DataFrame
            instances = [instance for instance in os.listdir(wid_path) if instance[-5:] == '.JPEG']
            for instance in instances:
                    dataframe.loc[(phase, wid, instance)] = synset

            print(f'{phase}: {num}')
                    
    # Save the DataFrame for later use in constructing our DataSet and DataLoader objects
    dataframe.to_pickle(os.path.abspath(f'{output_path}{dataset}.pkl'))

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

    # Call a helper function to create, fill, and save a DataFrame with information about our data
    # Furniture
    # Self-Propelled Vehicles
    create_dataframe(image_directory, data_directory, dataset)
