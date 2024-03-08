import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer, required



##### python libraries
import os
import random
import math
import json
import argparse
import copy
import time
import datetime
import numpy as np
import pandas as pd
import datetime

from pathlib import Path 

import warnings 
warnings.filterwarnings('ignore')


### Single finetune
def single_finetune(exp,model,optimizer,criterion,train_loader,val_loader,epochs,\
                    save_dir_fine,ckt,i,device,train_s,ds,last,opt):

    
    hist = {'train':[], 'val':[]}
    accur= {'train':[], 'val':[]}    
    best_loss = 1000
    
    for epoch in range(epochs):
        
        tic = time.time()
            
        for phase in ['train','val']:
            if phase=='train':
                model.train()
                data_loader = train_loader
                    
            else:
                model.eval()
                data_loader = val_loader
                                                   
            total_loss = 0
            correct = 0
            
            for batch_idx,(img,lable) in enumerate(data_loader):
                
                img = img.to(device)
                lable = lable.to(device)
                
                logits = model(img)
                loss = criterion(logits,lable)
                
                if phase=='train':
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                total_loss += loss.item() 
                pred = logits.argmax(1)
                    
                correct += (pred==lable).sum().item()
                
                        
            error = total_loss/len(data_loader)
            acc = correct/len(data_loader.dataset)  
            
            hist[phase].append(error)
            accur[phase].append(acc)
                   
        toc = time.time()
        time_epoch = toc - tic
                
        print('Epoch: %d Train_Loss: %0.4f, Val_Loss: %0.4f,  Train_Acc: %0.4f, Val_Acc: %0.4f , time:%0.4f seconds'%\
              (epoch,hist['train'][epoch],hist['val'][epoch],accur['train'][epoch],accur['val'][epoch],time_epoch))
            
            
        ## saving checkpoints when val loss is minimum
        is_best = bool(hist['val'][epoch] < best_loss)
        best_loss = hist['val'][epoch] if is_best else best_loss
        loss_val = hist['val'][epoch]
        
        if is_best:
            
            checkpoints = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': loss_val}
            torch.save(checkpoints,os.path.join(save_dir_fine,f'{exp}_{train_s}%_{ckt}_{i}_best_model.pt'))
                
    ## saving last checkpoint
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': hist['val'][epoch]},\
                os.path.join(save_dir_fine,f'{exp}_{train_s}%_{ckt}_{i}_last_model.pt'))
