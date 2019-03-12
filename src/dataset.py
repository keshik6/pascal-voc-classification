# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:23:51 2019

@author: Keshik
"""

from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
import math
import utils

class PascalVOC_Dataset(Dataset):
    """
    PascalVOC Dataset class extends the pytorch dataset

    Attributes:
        image_dir (str): path to where the images are stored
        labels_df (str): path to the csv file where the labels are stored
        transforms (obj): pytorch transformations
        fraction (float): fraction of the original dataset to use. 0.0 < fraction <= 1.0

    """
    
    
    """
    Initialize class attributes

    Args:
        image_dir (str): path to where the images are stored
        labels_df (str): path to the csv file where the labels are stored
        transforms (obj): pytorch transformations
        fraction (float): fraction of the original dataset to use
    Raises:
        FileNotFoundError: If the labels_df csv file does not exist
    """
    def __init__(self, image_dir, labels_df, transforms, fraction = 1.0):
        if not (os.path.isdir(image_dir)) or not (os.path.isfile(labels_df) ):
            raise FileNotFoundError
        self.labels_df = labels_df
        self.image_dir = image_dir
        self.transforms = transforms
        self.fraction = fraction
        self.labels_df = pd.read_csv(labels_df, nrows = math.ceil(self.fraction * utils.get_nrows(self.labels_df)))
    
    
    """
    Getitem from dataset

    Args:
        index: index of the file
    Raises:
        FileNotFoundError: If the target image file does not exist
    Returns:
        image, label (20 row numpy vector)
    """
    def __getitem__(self, index):
        # Get image name
        img_name = self.labels_df.loc[index, "Image"]
        label = np.asarray(self.labels_df.iloc[index, :][1:])
        
        # Read image
        img_path = "{}.jpg".format(os.path.join(self.image_dir, img_name))
        
        if not os.path.isfile(img_path):
            raise FileNotFoundError
        
        # Open the image now
        image = Image.open(img_path)
        
        # Convert grayscale images to rgb
        if (image.mode != "RGB"):
            image = image.convert(mode = "RGB")
            
        if self.transforms is not None:
            image = self.transforms(image)
            return image, label
        
        return image, label
        
    
    """
    Get size of dataset

    Returns:
        size of the dataset
    """
    def __len__(self):
        return math.ceil(self.fraction * self.labels_df.shape[0])
    

# Test your functions here
transforms = transforms.Compose([transforms.Resize(224), 
                    transforms.CenterCrop(224), 
                    transforms.ToTensor()])

PascalDataSet = PascalVOC_Dataset("../data/JPEGImages/", "../data/labels.csv", transforms=transforms, fraction=1)
img, label = PascalDataSet.__getitem__(500)
print(img, label)
