import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import lr_scheduler
#### pakckages from totrchvison
import torchvision
from torchvision import datasets,transforms
import torchvision.models as models

##### python libraries
import os, glob
import random
import math
import json
import seaborn as sns
import argparse
import time
import datetime
import collections
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score
from pathlib import Path 
from sklearn.model_selection import ShuffleSplit

from networks import networkbyol, finetune_net
try:
    import cPickle as pickle
except:
    import pickle    
import warnings 
warnings.filterwarnings('ignore')

######## Preprocess 
def preprocess(ds,in_size=None):
    if ds == 'svhn' or ds == 'cifar100' or ds == 'cifar10':
        transform=transforms.Compose([transforms.Resize(size=(in_size,in_size)),
                              transforms.ToTensor()])
    elif ds == 'lsun':
        transform=transforms.Compose([transforms.Resize(size=(in_size,in_size)),
                              transforms.ToTensor()])
    return(transform)

######## Computing uncertainties
def Entropy(p):
    """ p :[N_data,num_cls]
        H: [N_data,num_cls]
    """
    H = -(p *np.log(p)).sum(axis=1)
    # meanH: mean entropy on whole dataset 
    meanH = H.mean(axis=0)
    stdH = H.std(axis=0)
    return(H,meanH,stdH)

######## evaluation metric for OOD detection (cdf)
def cdf(ent,num_cls_OOD):
    p = 1/num_cls_OOD
    max_ent = - num_cls_OOD * p *np.log(p)    
    up_bound = max_ent.round(1)+0.3
    unc_thr = [l.round(2) for l in list(np.arange(0.0,up_bound,0.1))]
    emp_cdf = dict.fromkeys(unc_thr)
    for thr in unc_thr:
        ent_thr = ent[ent<=thr]
        emp_cdf[thr] = ent_thr.sum() / ent.sum()
    return(emp_cdf)
   
######## get AUC score
def get_auroc(pr_pos, pr_neg):
    # pos: InD with labels 1
    # neg: OOD with labels 0
    
    pos = pr_pos[:].reshape((-1, 1))
    neg = pr_neg[:].reshape((-1, 1))
   
    # pr of InD and OOD samples
    samples = np.vstack((pos, neg))
    labels = np.zeros(len(samples), dtype=np.int32)
    labels[:len(pos)] += 1    
    roc_auc_correct = roc_auc_score(labels, samples)    
    return(round(roc_auc_correct*100,1))

##############################

# The main execution 
parser = argparse.ArgumentParser(description='OOD test of Byol and BByol')

parser.add_argument('--seed',type=int,default=42,
                    help='the seed for experiments!')

parser.add_argument('--exp',type=int,default=770,
                        help='ID of expriment that we want to use for OOD!')

parser.add_argument('--opt',type=str,choices=('adam','sgd','sghm'), default='sghm',
                       help='optimizer to be used!')

parser.add_argument('--prtr',type=bool, default=False,
                       help='If we want to use pretrained resnet or not!')

parser.add_argument('--model',type=str,default='cbyol',choices=('byol','bbyol','cbyol'),
                        help='which model we want to use for OOD!')

# model information 
parser.add_argument('--emb_size',type=int, choices=(512,4096), default=512,
                       help='embeding size!')

parser.add_argument('--proj_size',type=int, default=128,
                       help='the size of projection map')

# data and datasets
parser.add_argument('--in_size',type=int,default=96,
                        help='The input size of images in pretrained model!')

parser.add_argument('--split',nargs="+", default=100,
                       help='the split of train set that we want to consider as finetuned model!')

parser.add_argument('--ds_pr',type=str,default='stl10', choices=('cifar10','cifar100','stl10'),
                        help='Dataset that we pretrained the model!')

parser.add_argument('--ds_ft',type=str,default='cifar100', choices=('cifar10','cifar100','stl10'),
                        help='Dataset for loading finetune model!')

parser.add_argument('--ood_ds',type=str,default='cifar10', choices=('svhn','lsun','cifar100','cifar10'),
                        help='Dataset that we want to use for OOD!') 

parser.add_argument('--num_classes',type=int, choices=(10,100), default=100,
                       help='The number of classes in fine tunned model')

parser.add_argument('--num_cls_OOD',type=int, choices=(10,100), default=10,
                       help='The number of classes in OOD!')

parser.add_argument('--b_size',type=int, default=100,
                       help='batch size for testset.')

# number of ensembles 
parser.add_argument('--burn_in',type=int,default=5,
                        help='If we want to consider any checkpoint as burn-in phase!')

parser.add_argument('--num_ens',type=int, default=6,
                       help='the number of ensembles that we want to take!')


parser.add_argument('--burn_in_ind',type=int,default=1,
                        help='The burn in that we used for in distribution finetunning!')

parser.add_argument('--in_dist',type=bool,default=True,
                        help='If we want to have in_dist analysis!')

# plotting
parser.add_argument('--plot',type=bool, default=True,
                       help='If we want to plot cdf or pdf histograms!')

parser.add_argument('--write_exp',type=bool, default=True,
                       help='If we want to write our expriments in csv file or not!')

args = parser.parse_args()



def OOD(args):
    
    save_dir = Path('./')
    save_dir_fine = save_dir / 'byol_ckpts' / 'finetune' / args.ds_ft
    result_dir = save_dir / 'results' / 'OOD'
    
    # setting the device    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    
    # defining loss function
    criterion = nn.CrossEntropyLoss().to(device)

    backbone = models.resnet18(pretrained=args.prtr, progress=True)
    
    # making ood datasets and data loaders
    if args.ood_ds=="svhn":
        svhn = datasets.SVHN(root='./',split='test',transform=preprocess(args.ood_ds,args.in_size),download=True)
        testloader = DataLoader(dataset=svhn,batch_size=args.b_size,shuffle=False,num_workers=4,drop_last=True)
                    
    elif args.ood_ds=='cifar100':
        cifar100 = datasets.CIFAR100('./data',train=False,transform=preprocess(args.ood_ds),download=True)
        testloader = DataLoader(dataset=cifar100,batch_size=args.b_size,shuffle=False,num_workers=4,drop_last=True)
        
    elif args.ood_ds=='cifar10':
        cifar10 = datasets.CIFAR10('./data',train=False,transform=preprocess(args.ood_ds,args.in_size),download=True)
        testloader = DataLoader(dataset=cifar10,batch_size=args.b_size,shuffle=False,num_workers=4,drop_last=True)
            
    tot_ens = args.num_ens - args.burn_in 
    tot_output = []
    tot_gt = []
    mean_pr_logits = []
    ce_loss = 0
    nll = 0
    correct = 0
    
    for j,(img,lable) in enumerate(testloader):
        
        img = img.to(device)
        lable = lable.to(device)
        tot_gt+=list(lable.data.cpu().numpy())
        batch_output = img.data.new(tot_ens,len(img),args.num_cls_OOD)
            
        for idx,i in enumerate(range(args.burn_in,args.num_ens)):         
            net = networkbyol('online',backbone,args.emb_size,args.proj_size).to(device)
            encoder = nn.Sequential(*list(net.children())[:-2])
            model = finetune_net(encoder,args.emb_size,args.num_classes).to(device)
                
            # load best model from fine tunned modesl dir
            ckt_inf = torch.load(os.path.join(save_dir_fine,f'{args.exp}_{args.split}%_all_{i}_best_model.pt'),\
                                 map_location=device)
            
            model.load_state_dict(ckt_inf['model'])
            if args.num_classes != args.num_cls_OOD:
                model.linear = nn.Linear(args.emb_size,args.num_cls_OOD).to(device)
            batch_output[idx] = model(img.float())
        
        # take mean of logits and compute CE loss: [batch,cls_OOD]
        mean_logits_batch = batch_output.mean(dim=0)
        loss1 = criterion(mean_logits_batch,lable)  # reduction:mean
        ce_loss += loss1.item()
        
        pred = mean_logits_batch.argmax(1)
        correct += (pred==lable).sum().item()
        
        # pr_logits_batch: [n_ens,batch,cls]
        # mean_pr_batch: [batch,cls_OOD]
        mean_pr_logits_batch = F.softmax(batch_output,dim=2).mean(dim=0,keepdim=False)
        log_mean_pr_batch = torch.log(mean_pr_logits_batch)
        loss2 = F.nll_loss(log_mean_pr_batch,lable) # reduction:mean
        nll += loss2.item()
        mean_pr_logits.append(mean_pr_logits_batch.data.cpu().numpy())
                
    ce_loss /= len(testloader)
    nll /= len(testloader) # this nll is wrong 
    acc = correct / len(testloader.dataset) 

    mean_pr_logits = np.concatenate(mean_pr_logits)
    epistemic, mean_eps, std_eps = Entropy(mean_pr_logits)
    
    if args.in_dist:    
        pr_tot = np.load(os.path.join(result_dir_exp,'burn_%d_pr_ens.npy'%(args.burn_in_ind)))
        ig_sam = pr_tot.shape[0] - tot_ens
        log_tot = np.load(os.path.join(result_dir_exp,'burn_%d_logits_ens.npy'%(args.burn_in_ind)))
        mean_pr_InD = pr_tot[ig_sam:].mean(axis=0)
        epistemic_InD, mean_eps_InD, std_eps_InD = Entropy(mean_pr_InD)
        
    tot_gt = np.array(tot_gt)
    
    # pos: IND => label: 1
    # neg: OOD => label: 0
    pr_pos = mean_pr_InD.max(axis=1)
    pr_neg = mean_pr_logits.max(axis=1)
    
    auc_roc_scr = get_auroc(pr_pos,pr_neg)
    
    if args.write_exp: 
        data = {
            'ood_ds':[args.ood_ds],
            'ssl':[args.model],
            'exp': [args.exp],
            'ds' :[args.ds_ft],
            'split':[args.split],
            'bur':[args.burn_in],
            'n_ens': [tot_ens],
            'NLL':[round(nll,1)],
            'CE':[round(ce_loss,1)],
            'ACC':[round(acc,1)],
            'AUC_ROC':[auc_roc_scr],
            'mean_H':[round(mean_eps,1)],
            'std_H':[round(std_eps,1)]
        }
        
        # ./results
        csv_path = Path(os.path.join(result_dir,'run_sweeps_test_OOD.csv'))

        if os.path.exists(csv_path): 
            sweeps_df = pd.read_csv(csv_path)
            sweeps_df = sweeps_df.append(
            pd.DataFrame.from_dict(data), ignore_index=True).set_index('exp')

        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True) 
            sweeps_df = pd.DataFrame.from_dict(data).set_index('exp')

        # save experiment metadata csv file
        sweeps_df.to_csv(csv_path)
                
# main execuation
if __name__=='__main__':
   
    OOD(args)
    
    
            
            
            
            
         
            
                
                
                
                
                                 











