# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:50:25 2019

@author: Keshik
"""

import torch
import numpy as np
from torchvision import transforms
import torchvision.models as  models
from torch.utils.data import DataLoader
from dataset import PascalVOC_Dataset
import torch.optim as optim
from train import train_model, test
from utils import encode_labels, plot_history
import os
import torch.utils.model_zoo as model_zoo

def main(data_dir, model_name, lr, epochs, batch_size = 32):
    
    model_dir = os.path.join("../models", model_name)
    
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
    model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }
    
    model_collections_dict = {
            "resnet18": models.resnet18(),
            "resnet34": models.resnet34(),
            "resnet50": models.resnet50()
            }
    
    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print("Available device = ", device)
    model = model_collections_dict[model_name]
    model.load_state_dict(model_zoo.load_url(model_urls[model_name]))
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 20)
    model.to(device)
           
    optimizer = optim.SGD([   
            {'params': list(model.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
            {'params': list(model.parameters())[-1], 'lr': lr[1], 'momentum': 0.9, 'weight_decay': 1e-4}
            ])
    
    
    # Imagnet values
    mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
    
    transformations = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(p = 0.25),
                                      transforms.ColorJitter(brightness=0.15),
                                      transforms.RandomRotation(30),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean = mean, std = std),
                                      ])
    
    transformations_valid = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean = mean, std = std),
                                          ])

    # Create train dataloader
    dataset_train = PascalVOC_Dataset(data_dir,
                                      year='2012', 
                                      image_set='train', 
                                      download=False, 
                                      transform=transformations, 
                                      target_transform=encode_labels)
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=4)
    #print(len(train_loader.dataset))
    
    # Create validation dataloader
    dataset_valid = PascalVOC_Dataset(data_dir,
                                      year='2012', 
                                      image_set='val', 
                                      download=False, 
                                      transform=transformations_valid, 
                                      target_transform=encode_labels)
    
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=4)
    #print(len(valid_loader.dataset))
    
    # Load the best weights before testing
    print("Loading best weights")
    if os.path.isfile(os.path.join(model_dir, "model")):
        model.load_state_dict(torch.load(os.path.join(model_dir, "model")))
    
    
    log_file = open(os.path.join(model_dir, "log.txt"), "w+")
    log_file.write("----------Experiment - {}-----------\n".format(model_name))
    trn_hist, val_hist = train_model(model, device, optimizer, train_loader, valid_loader, model_dir, epochs, log_file)
    torch.cuda.empty_cache()
    
    plot_history(trn_hist[0], val_hist[0], os.path.join(model_dir, "loss"))
    plot_history(trn_hist[1], val_hist[1], os.path.join(model_dir, "accuracy"))
#    np.save(os.path.join(directory,"train_hist"), np.asarray(trn_hist[:2]))
#    np.save(os.path.join(directory,"val_hist"), np.asarray(val_hist[:2]))
    
    #---------------Test your model here---------------------------------------
    # Load the best weights before testing
    print("Evaluating model on test set")
    print("Loading best weights")
    model.load_state_dict(torch.load(os.path.join(model_dir, "model")))
    transformations_test = transforms.Compose([transforms.Resize(256), 
                                          transforms.FiveCrop(224), 
                                          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                          transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean = mean, std = std)(crop) for crop in crops])),
                                          ])
    
    dataset_test = PascalVOC_Dataset(data_dir,
                                      year='2012', 
                                      image_set='val', 
                                      download=False, 
                                      transform=transformations_test, 
                                      target_transform=encode_labels)
    
    test_loader = DataLoader(dataset_test, batch_size=int(batch_size/5), num_workers=0)
    test(model, device, test_loader)
    
    log_file.close()

if __name__ == '__main__':
    main('../data/', "resnet50", lr = [1e-4, 5e-3], epochs = 30)