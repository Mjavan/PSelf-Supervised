# /core/split_datasets.py
import torch
from torch.utils.data import Dataset, DataLoader, Subset

#### pakckages from totrchvison
import torchvision
from torchvision import datasets,transforms


##### python libraries
import os
import random
import math
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import ShuffleSplit

try:
    import cPickle as pickle
except:
    import pickle

import warnings 
warnings.filterwarnings('ignore')


#### Preprocessing & Data augmentation #####
def get_transform(in_size,ds,phase='train'):
    
    if ds == 'cifar10':
        mean_ds = [0.491, 0.482, 0.447]
        std_ds = [0.247, 0.243, 0.261]
        
    elif ds == 'cifar100':
        mean_ds = [0.507, 0.486, 0.440]
        std_ds = [0.267, 0.256, 0.276]
        
    elif ds == 'mnist':
        mean_ds = (0.1307,)
        std_ds = std=(0.3081,)
        
    elif ds == 'stl10':
        mean_ds = [0.485, 0.456, 0.406]
        std_ds = [0.229, 0.224, 0.225]
        
    elif ds == 'imagenet10' or ds == 'tinyimagenet':
        mean_ds = [0.485, 0.456, 0.406]
        std_ds = [0.229, 0.224, 0.225]
           
    if phase =='train':
        transform=transforms.Compose([transforms.RandomResizedCrop(size=in_size),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean_ds,std_ds)])     
    else:
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean_ds,std_ds)])  
    return(transform)


##### Data loaders ####### 
def get_train_val_set(dataset,ds,val_size,in_size,seed,val2=0):
    
    ## making transformations
    transform_train = get_transform(in_size,ds,phase='train')
    transform_val = get_transform(in_size,ds,phase='val')
    split = ShuffleSplit(n_splits=1,test_size=val_size,random_state=seed)
    for train_idx, val_idx in split.split(range(len(dataset))):
        train_index= train_idx
        val_index = val_idx
        
    train_set1 = Subset(dataset,train_index)
    val_set1 = Subset(dataset,val_index)
    
    if not val2:        
        train_set1.transform = transform_train
        val_set1.transform = transform_val
        return(train_set1,val_set1)
        
    val2 = val_size*0.1
    split = ShuffleSplit(n_splits=1,test_size=val2,random_state=seed)
    for train_idx, val_idx in split.split(range(len(train_set1))):
        train_index= train_idx
        val_index = val_idx
        
    train_set2 = Subset(train_set1,train_index)
    val_set2 = Subset(train_set1,val_index)
        
    val_set1.transform = transform_train
    val_set2.transform = transform_val
    return(val_set1,val_set2)
          
############## making dataloaders and saving them ##############

parser = argparse.ArgumentParser(description='Making datasets')

parser.add_argument('--seed',type=int,default=42,
                    help='the seed for experiments!')

parser.add_argument('--ds',type=str,default='imagenet10',
                    choices=('cifar10','cifar100','mnist','stl10','imagenet10','tinyimagenet'),
                    help='the dataset for finetunning!')

parser.add_argument('--in_size',type=int,default=224,choices=(32,28,64,96,224),
                       help='the input size of images')

parser.add_argument('--splits',nargs="+",type=int, default=[100],
                       help='the split of train set that we want to have!')

args = parser.parse_args()

def split_datasets(args):
    
    save_dir  = Path('./')
    
    svd_split = save_dir / 'data' / 'split_data'
    os.makedirs(save_split, exist_ok=True)
    
    ## loading data sets 
    if args.ds == 'cifar10':
        transform=transforms.Compose([transforms.Resize(size=args.in_size),transforms.ToTensor()])
        trainset = datasets.CIFAR10('./data',train=True,transform=transform,download=True)
        
    elif args.ds == 'cifar100':
        transform=transforms.Compose([transforms.Resize(size=args.in_size),transforms.ToTensor()])
        trainset = datasets.CIFAR100('./data',train=True,transform=transform,download=True)
        
    elif args.ds == 'stl10':
        transform=transforms.Compose([transforms.Resize(size=args.in_size),transforms.ToTensor()])
        trainset = datasets.STL10('./data',split='train',transform=transform,download=True)
        
    elif args.ds == 'imagenet10':
        path_imagenet = save_dir/'data'/'imagenet10'/'train'
        transform=transforms.Compose([transforms.Resize(size=(args.in_size,args.in_size)),transforms.ToTensor()])       
        trainset = datasets.ImageFolder(path_imagenet,transform=transform)
        
    elif args.ds == 'tinyimagenet':
        path_tinyimagenet = save_dir/'data'/'tiny-imagenet-200'/'train'
        transform=transforms.Compose([transforms.Resize(size=(args.in_size,args.in_size)),transforms.ToTensor()])
        trainset = datasets.ImageFolder(path_tinyimagenet,transform=transform)
                                            
    for split in args.splits:
        train_split= split / 100
        print(f'train_split is:{train_split}')
        
        if split==100:
            val2=0 
            val_size=0.1
        else:
            val2=1
            val_size=train_split
        
        train_set, val_set = get_train_val_set(trainset,args.ds,val_size=val_size,in_size=args.in_size,\
                                               seed=args.seed,val2=val2)    
    
        ## Saving train_set & val_set
        save_train = svd_split / f'train_set_{args.ds}_{args.in_size}_{split}%.pickle'
        save_val = svd_split / f'val_set_{args.ds}_{args.in_size}_{split}%.pickle'
        
        with open(save_train, 'wb') as f:
            pickle.dump(train_set,f,protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(save_val, 'wb') as f:
            pickle.dump(val_set,f,protocol=pickle.HIGHEST_PROTOCOL)
        
    print('Making datasets were done!')
    return(train_set,val_set)
    
#### running script ####
if __name__=='__main__':
    split_datasets(args)
    

    
    
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    









