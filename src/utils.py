# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:52:33 2019

@author: Keshik
"""
import pandas as pd
import os


labels_root_dir = "../data/ImageSets/Main/"
labels_save_dir = "../data/"


def get_categories(labels_dir):
    """
    Get the object categories
    
    Args:
        label_dir: Directory that contains object specific label as .txt files
    Raises:
        FileNotFoundError: If the label directory does not exist
    Returns:
        Object categories as a list
    """
    
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError
    
    else:
        categories = []
        
        for file in os.listdir(labels_dir):
            if file.endswith("_train.txt"):
                categories.append(file.split("_")[0])
        
        return categories


def encode_labels(labels_dir, categories, save_location):
    """
    Create csv file to encompass all labels for images 
    
    Args:
        label_dir: Directory that contains object specific label as .txt files
    Raises:
        FileNotFoundError: If the labels file corresponding to an object category does not exist
    Returns:
        None
    """
    
    df = pd.DataFrame()
    
    for i in range(len(categories)):
        
        file_name = os.path.join(labels_dir, "{}_train.txt".format(categories[i]))
        
        if not os.path.isfile(file_name):
            raise FileNotFoundError
        else:
            if i == 0:
                df = pd.read_fwf(file_name, index=False, names=["Image", categories[i]])
            else:
                df_temp = pd.read_fwf(file_name, index=False, names=["Image", categories[i]])
                df = df.merge(df_temp, how="right", on=["Image"])
            
    df.to_csv(os.path.join(save_location, 'labels.csv'), index=False)


def get_nrows(file_name):
    """
    Get the number of rows of a csv file
    
    Args:
        file_path: path of the csv file
    Raises:
        FileNotFoundError: If the csv file does not exist
    Returns:
        number of rows
    """
    
    if not os.path.isfile(file_name):
        raise FileNotFoundError
    
    s = 0
    with open(file_name) as f:
        s = sum(1 for line in f)
    return s


categories = get_categories(labels_root_dir)
encode_labels(labels_root_dir, categories, labels_save_dir)

