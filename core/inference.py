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


#### Testing on test set 
def inference(model,test_loader,criterion,device):
    
    model.eval()    
    test_loss =0     
    correct = 0
    out_pr = []
    gt_list = []
    logits = []
    
    with torch.no_grad():
        for batch_idx,(img,lable) in enumerate(test_loader):
            img = img.to(device)
            lable = lable.to(device)
            output = model(img)
            out_pr.append(F.softmax(output,dim=1).data.cpu().numpy())
            logits.append(output.data.cpu().numpy())
            gt_list += list(lable.data.cpu().numpy())
            loss = criterion(output,lable)
            test_loss += loss.item()
            pred = output.argmax(1)
            correct += (pred==lable).sum().item()
        error = test_loss / len(test_loader)        
        acc = correct / len(test_loader.dataset)
        out_pr = np.concatenate(out_pr)
        logits = np.concatenate(logits)
    print(f'Test_Loss: {error:0.4f}, Test_Accuracy:{acc:0.4f}\n')
    return(out_pr,logits,gt_list,error,acc)
