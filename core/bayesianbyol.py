# /core/bayesianbyol.py

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer, required
import torchvision
from torchvision import datasets,transforms
import torchvision.models as models

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


from networks import networkbyol
from dataloader import get_transform, pair_aug, get_train_val_loader
from csghmc import SGHM
from lr_scheduler import update_lr, update_lin 

import warnings 
warnings.filterwarnings('ignore')

#### Main training
parser = argparse.ArgumentParser(description='BByol pretraining')

parser.add_argument('--seed',type=int,default=42,
                    help='The seed for experiments!')

parser.add_argument('--exp',type=int,default=538,
                        help='ID of this expriment!')

parser.add_argument('--ds',type=str,default='stl10',choices=('stl10','tinyimagenet'),
                        help='Dataset for pretraining!')

parser.add_argument('--num_epochs',type=int, default=10,
                       help='Number of epoch for pretraining')

parser.add_argument('--model_type',type=str, default='cb', choices=('','b','c','cb'),
                       help='If we want to train byol, bbyol, cbyol, cbbyol!')

# model
parser.add_argument('--model_depth',type=str,choices=('res18','res50','res100'),default='res18',
                        help='Model to use as feature extractor!')

parser.add_argument('--mlp_hid_size',type=int, choices=(512,2048), default=512,
                       help='The mid size in MLP')

parser.add_argument('--proj_size',type=int, choices=(128,256,2048), default=128,
                       help='The size of projection map')

# optimizer 
parser.add_argument('--optimizer',type=str, default='sghm', choices=('adam','sgd','sghm'),
                       help='The optimizer to use')

parser.add_argument('--base_target_ema',type=float, default=0.996,
                       help='The size of projection map')

parser.add_argument('--temp',type=float, default=0.1, 
                        help = 'Temprature for cold posterior in sghm opt!')
# data
parser.add_argument('--in_size',type=int, default=32,
                       help='The input size of images')

parser.add_argument('--s',type=float,default=1.0,choices=(0.5,1.0),
                        help='The strength of color distortion! (s=0.5 for cifar10 and 100)')

parser.add_argument('--aug',type=bool,default=True,
                        help='If we want to have data augmentation or not!')

# dataloader
parser.add_argument('--batch_size',type=int, default=256,
                       help='The number of batch size for training')

parser.add_argument('--val_size',type=float, default=0.05,
                       help='The validation size')

parser.add_argument('--num_workers',type=int, default=4,
                       help='num_workers')

# lr & lr scheduler
parser.add_argument('--lr',type=float, default=2e-1,
                       help='learning rate')

parser.add_argument('--lr_sch',type = str, default='cyc',choices=('cyc','fixed','lin'), 
                        help = 'If we want to use a fixed or cyc leraning schedule')

parser.add_argument('--lr_dec',type=float,default=3.0,
                       help='lr decay for linear schedule!')

parser.add_argument('--cycle_length',type=int, default=50,
                       help='Number of epochs in each cycle to either lr update or save checkpoints!')

# regularizer
parser.add_argument('--wd',type=float,default=1,choices=(0,1,0.1,0.075,0.05,0.01,25),
                       help='weight decay')

parser.add_argument('--clip_grad',type = bool, default=False, 
                        help = 'If we want to clip grad or not!')

# inject noise & saving cechpoints
parser.add_argument('--epoch-noise',type =int, default=45, 
                        help = 'The epoch that we want to inject the noise, (set 0 if we do not want to inject noise)!')

parser.add_argument('--save_sample',type =bool, default =True, 
                        help = 'If we want to save samples or not!')

parser.add_argument('--epoch-st',type =int, default=350, 
                        help = 'The epoch that we want to start saving checkpints!')

parser.add_argument('--n_sam_cycle',type=int, default=1,
                       help='Number of samples in each cycle')

parser.add_argument('--N_samples',type =int, default=13, 
                        help = 'Number of sample weights that we want to take!')

parser.add_argument('--scale',type =bool, default =False, 
                        help = 'If we want to scale the loss or not!')
                        
args = parser.parse_args() 


def main(args):
    
    seed =args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
  
    ## setting device 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    
    ## making directory to save checkpoints 
    save_dir = Path('./')
    save_dir_ckpts = save_dir /'ckpts'/ 'byol_ckpts'
    save_dir_epoch = save_dir_ckpts / 'epoch_samples' / args.ds
    save_dir_mcmc = save_dir_ckpts / 'mcmc_samples'/ args.ds / f'{args.optimizer}_{args.exp}' 
    os.makedirs(save_dir_mcmc, exist_ok=True)
    save_dir_param = save_dir/'params'/'byol_params'/'pretrain'/args.ds
    os.makedirs(save_dir_param, exist_ok=True)
    
    ## saving version packages
    #os.system(f'conda env export > {save_dir}/yml/{args.optimizer}_{args.exp}_env.yml')
  
    ## saving hyperparameters
    HPS = vars(args)    
    with open(save_dir_param / f'{args.exp}_{args.model_type}param.json','w') as file:    
        json.dump(HPS,file,indent=4)
        
    ## getting dataset and dataloaders
    if args.ds == 'cifar10':
        dataset = datasets.CIFAR10('./data',train=True,transform=pair_aug(get_transform(args.in_size,args.ds,args.s,args.aug)),\
                                   download=True)
        
    elif args.ds == 'cifar100':
        dataset = datasets.CIFAR100('./data',train=True,transform=pair_aug(get_transform(args.in_size,args.ds,args.s)),\
                                    download=True)
        
    elif args.ds == 'imagenet10':
        path_imagenet = save_dir/'data'/'imagenet10'/'train'     
        dataset = datasets.ImageFolder(path_imagenet,transform=pair_aug(get_transform(args.in_size,args.ds,args.s,args.aug)))
             
    elif args.ds == 'stl10':
        dataset = datasets.STL10('./data', split='unlabeled',transform=pair_aug(get_transform(args.in_size,args.ds,args.s)),\
                                 download = True)
        
    elif args.ds == 'tinyimagenet':
        path_tinyimagenet = save_dir/'data'/'tiny-imagenet-200'/'train'
        dataset = datasets.ImageFolder(path_tinyimagenet,transform=pair_aug(get_transform(args.in_size,args.ds,args.s)))
                    
    train_loader,val_loader = get_train_val_loader(dataset,val_size=args.val_size,batch_size=args.batch_size,\
                                                   num_workers=args.num_workers,seed=args.seed)
    
    ## Setting parameters for cyclic learning rate schedule
    N_train = len(train_loader.dataset)
    n_batch = len(train_loader)
    cycle_batch_length = args.cycle_length * n_batch
    batch_idx = 0

    ## getting models 
    if args.model_depth=="res18":
        backbone = models.resnet18(pretrained=False, progress=True)
        
    elif args.model_depth=='res50':
        backbone = models.resnet50(pretrained=False, progress=True)
    
    online_network = networkbyol('online',backbone,args.mlp_hid_size,args.proj_size).to(device)
    target_network = copy.deepcopy(networkbyol('target',backbone,args.mlp_hid_size,args.proj_size)).to(device)
    
    ## getting optimizer
    if args.optimizer=='adam':
        optimizer = optim.Adam(online_network.parameters(),args.lr,weight_decay=args.wd)
        
    elif args.optimizer=='sgd':
        optimizer = optim.SGD(online_network.parameters(),lr=args.lr,weight_decay=args.wd,momentum=0.9)
        
    elif args.optimizer=='sghm':
        optimizer = SGHM(params=online_network.parameters(),lr=args.lr,weight_decay=args.wd/N_train,momentum=0.9,\
                         temp=args.temp,addnoise=1,dampening=0.0,N_train=N_train)    
        
    # initilizing target_network
    for online_params, target_params in zip(online_network.parameters(), target_network.parameters()):
            target_params.data.copy_(online_params.data)  # initialize
            target_params.requires_grad = False
    
    history = {'train':[], 'val':[]}
    best_val = float('inf')
    weight_set_samples = []
    sampled_epochs = []
    mt = 0 
    
    print(f'training is started')
    for epoch in range(args.num_epochs):
        tic = time.time()
        for phase in ['train','val']:
            
            if phase=='train':
                #print(f'############################')
                #print(f'# We are in training phase #')
                #print(f'############################')
                online_network.train()
                target_network.train()
                dataloader = train_loader
                
            else:
                
                #print(f'##############################')
                #print(f'# We are in validation phase #')
                #print(f'##############################')
                online_network.eval()
                target_network.eval()
                dataloader= val_loader
                
            total_loss = 0
            
            for (img1,img2),_ in dataloader:
                
                img1= img1.to(device)
                img2= img2.to(device)
                
                loss = Train(online_network,target_network,optimizer,img1,img2,phase,\
                             N_train,batch_idx,cycle_batch_length,epoch)             
                total_loss += loss.item()
                
                if phase=='train':    
                    batch_idx+=1
                            
            history[phase].append(total_loss/len(dataloader))
            
            if args.save_sample:
            
                if epoch>=args.epoch_st and (epoch%args.cycle_length)+1>(args.cycle_length-args.n_sam_cycle) and phase=='train':

                    sampled_epochs.append(epoch)
                    if use_cuda:
                        online_network.cpu()
                    torch.save(online_network.state_dict(),os.path.join(save_dir_mcmc,f'model_{mt}.pt'))
                    mt +=1
                    online_network.cuda()        
                    print(f'sample {mt} from {args.N_samples} was taken!')
                    print(f'sampled epoch lr:%.7f'%(optimizer.param_groups[0]['lr']))
                  
        toc = time.time()
        runtime_epoch = toc - tic
        lr_epoch = optimizer.param_groups[0]['lr']
        print('Epoch: %d Train_Loss: %0.4f, Val_Loss: %0.4f, time:%0.4f seconds, lr:%0.7f'%(epoch,history['train'][epoch],\
                                                                                  history['val'][epoch],
                                                                                  runtime_epoch,lr_epoch))                                                                  
        ### save best model    
        if history['val'][epoch] < best_val:
            best_val = history['val'][epoch]
            check_without_progress = 0 
            torch.save({'epoch':epoch+1,
                        'model':online_network.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'Loss':history['val'][epoch]},os.path.join(save_dir_epoch,f'{args.exp}_best_{args.model_type}byol.pt'))
          
    ### save last model
    torch.save({'epoch':epoch+1,
                'model':online_network.state_dict(),
                'optimizer':optimizer.state_dict(),
                'Loss':history['val'][epoch]},os.path.join(save_dir_epoch,f'{args.exp}_last_{args.model_type}byol.pt'))
    print(f'save last model')
                                                
def loss_fn(online_network,target_network,img1,img2,phase):
    with torch.set_grad_enabled(phase == 'train'):
        onl_pred1 = online_network(img1)
        onl_pred2 = online_network(img2)
    
        with torch.no_grad():
            tar_proj1 = target_network(img1)
            tar_proj2 = target_network(img2)
    
        def reg_loss(x,y):
            x = F.normalize(x, dim=1)
            y = F.normalize(y, dim=1)
            return(2-2 * (x * y).sum(dim=-1))
    
        loss = reg_loss(onl_pred1,tar_proj2)
        loss += reg_loss(onl_pred2,tar_proj1)
        return(loss.mean())
                   
def Train(online_network,target_network,optimizer,img1,img2,phase,N_train,batch_idx,cycle_batch_length,epoch):     
    loss = loss_fn(online_network,target_network,img1,img2,phase)
    if phase=='train':    
        # Update online network
        if args.scale:
            loss = loss * N_train    
        optimizer.zero_grad()
        # Update lr 
        if args.lr_sch == 'cyc':
            update_lr(args.lr,batch_idx,cycle_batch_length,args.n_sam_cycle,optimizer)
        if args.lr_sch == 'lin':
            lr = args.lr*np.exp(-args.lr_dec*min(1.0,(batch_idx*args.batch_size)/float(N_train)))
            update_lin(lr,optimizer)
        # Inject noise to parameter update
        if args.lr_sch == 'cyc' and args.epoch_noise:
            if (epoch%args.cycle_length)+1 > args.epoch_noise:
                optimizer.param_groups[0]['epoch_noise'] = True                       
            else:         
                optimizer.param_groups[0]['epoch_noise'] = False        
        elif args.lr_sch in ['lin','fixed'] and args.epoch_noise: 
            if epoch >= args.epoch_noise:
                optimizer.param_groups[0]['epoch_noise'] = True 
            else:
                optimizer.param_groups[0]['epoch_noise'] = False    
        loss.backward() 
        optimizer.step()
      
        # Update target network
        tau = args.base_target_ema
        for online_params,target_params in zip(online_network.parameters(),target_network.parameters()):
            target_params.data = tau*target_params +(1-tau)*online_params
        # Default is False
        if args.scale:
            loss = loss/N_train        
    return(loss)
  
#### Running the code
if __name__=='__main__':  
    main(args)
    
