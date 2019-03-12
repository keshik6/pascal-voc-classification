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

class PascalVOC_Dataset(Dataset):
    
    def __init__(self, image_dir, labels_df, transforms, fraction = 1.0):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_df)
        self.transforms = transforms
        self.fraction = fraction
    
    
    def __getitem__(self, index):
        # Get image name
        img_name = self.labels_df.loc[index, "Image"]
        label = np.asarray(self.labels_df.iloc[index, :][1:])
        
        # Read image
        img_path = "{}.jpg".format(os.path.join(self.image_dir, img_name))
        
        # Open the image now
        image = Image.open(img_path)
        
        # Convert grayscale images to rgb
        if (image.mode != "RGB"):
            image = image.convert(mode = "RGB")
            
        if self.transforms is not None:
            image = self.transforms(image)
            return image, label
        
        return image, label
        
    
    def __len__(self):
        return self.labels_df.shape[0]
    

transforms = transforms.Compose([transforms.Resize(224), 
                    transforms.CenterCrop(224), 
                    transforms.ToTensor()])
PascalDataSet = PascalVOC_Dataset("../data/JPEGImages/", "../data/labels.csv", transforms=transforms)
img, label = PascalDataSet.__getitem__(100)
print(img, label)