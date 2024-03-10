import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer, required



##### python libraries
import os, glob
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

from networks import networkbyol,networksimclr,finetune_net
from utils import evaluation_metrics
from dataloader import get_transform_ft as get_transform

try:
    import cPickle as pickle
except:
    import pickle
import warnings 
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Baysian Self-Supervised finetunning')

parser.add_argument('--seed',type=int,default=42,
                    help='the seed for experiments!')

parser.add_argument('--exp',type=int,default=477,
                        help='ID of this expriment!')

parser.add_argument('--cont',type=str,default='byol',choices=('byol','simclr'),
                        help='The contrastive model that we used for pretraining!')

parser.add_argument('--ds_pr',type=str,default='stl10',choices=('stl10','tinyimagenet'),
                        help='Dataset that we pretrained the model on that!')

parser.add_argument('--ds_ft',type=str,default='cifar10', 
                    choices=('cifar100','cifar10','stl10','imagenet10','tinyimagenet'),
                        help='Dataset for semisupervised learning or transfer learning!')

# Fine-tunning parameters
parser.add_argument('--fine',type=bool, default=False,
                       help='if we want to do finetunning or not!')

parser.add_argument('--ckt',type=str, default='all',choices=('best','last','all'),
                       help='if we want to use best, or last or MCMC checkpoints (all)')

parser.add_argument('--burn_in',type=int,default=0,
                        help='The burn in epochs!')

parser.add_argument('--ckt_sp',type=int,default=None,
                        help='The specific checkpoint that we want to look at!')

parser.add_argument('--ckt_inf',type=str,default='best',choices=('best','last'),
                       help='if we want to use best, or last model for inference')

# Optimizer & lr
parser.add_argument('--opt',type=str,default='sghm',choices=('adam','sgd','sghm'),
                       help='optimizer which was used in pretraining!')

parser.add_argument('--nes',type=bool, default=True,
                       help='if we want to have nestrov for sgd or not')

parser.add_argument('--wd',type=float, default=0,
                       help='Weight decay for finetunning!')

parser.add_argument('--num_epochs',type=int, default=50,
                       help='the number of epoch for pretraining')

parser.add_argument('--lr',type=float, default=2e-4,
                       help='learning rate')

# Architecure
parser.add_argument('--model_type',type=str, default='cb', choices=('','b','c','cb'),
                       help='If we want to train byol, bbyol, cbyol, cbbyol!')

parser.add_argument('--model_depth',type=str,choices=('res18','res50','res100'),default='res18',
                        help='model to use (causion with mlp_hidden_size, projection_size)')

parser.add_argument('--mlp_hid_size',type=int, choices=(512,4096), default=512,
                       help='the mid size in MLP')

parser.add_argument('--proj_size',type=int, choices=(128,256), default=128,
                       help='the size of projection map')

parser.add_argument('--num_classes',type=int, default=10,
                       help='the number of output classes in test set and semisupervised setting!')

# Test set and dataloaders
parser.add_argument('--num_workers',type=int, default=4,
                       help='num_workers')

parser.add_argument('--batch_size',type=int, default=80,
                       help='the number of batch size for training')

parser.add_argument('--exp_split',nargs="+", default=[100],
                       help='the split of train set that we want to consider')

# Evaluation parameters  
parser.add_argument('--pr_ens',type=bool, default=False,
                       help='if we want to save pr ensemble or not (when we are not in weight & biases)!')

parser.add_argument('--eval',type=bool, default=True,
                       help='if we want to evaluate our model or not!')

parser.add_argument('--ig_sam',type=int, default=1,
                       help='if we want to ignore samples from begining!')

parser.add_argument('--write_exp',type=bool, default=True,
                       help='if we want to write results in a csv file!')

args = parser.parse_args()

def semi_supervised(args):
    
    print(f'Baysian Semi-supervised evaluation')
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    
    
    ## setting device 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ## the directory to load checkpoints 
    save_dir  = Path('./')
    load_dir  = save_dir / 'ckpts' 
    split_dir = save_dir / 'data' / 'split_data' 
    exp_dir = load_dir / f'{args.cont}_ckpts' /'mcmc_samples'/ args.ds_pr 
    param_dir = save_dir /'params'/f'{args.cont}_params'/'finetune'/ args.ds_ft /\
    f'exp_{args.exp}_opt_{args.opt}_pr_{args.ds_pr}'
    param_dir_pr = save_dir/'params'/f'{args.cont}_params'/'pretrain'/args.ds_pr
    
    ## making a directory for results!
    result_dir_exp = save_dir / 'results' / f'{args.ds_ft}_{args.exp_split[0]}'/ f'{args.exp}_pr_{args.ds_pr}'

    os.makedirs(param_dir, exist_ok=True)
    os.makedirs(result_dir_exp,exist_ok=True)
    
    ## making a dic from hyperparameters
    config = vars(args) 
    
    ## loadig hyperparameters pretrained model
    with open(param_dir_pr/f'{args.exp}_{args.model_type}param.json') as file:  
        HPP = json.load(file)
        in_size = HPP['in_size']  
      
    ## loading test_set
    if config['ds_ft']=='cifar10':                          
        testset = datasets.CIFAR10('./data',train=False,\
                                   transform=get_transform(config['ds_ft'],in_size),download=True)
    
    elif config['ds_ft']=='cifar100':
        testset = datasets.CIFAR100('./data',train=False,\
                                    transform=get_transform(config['ds_ft'],in_size),download=True)
        
    elif config['ds_ft']=='stl10':
        testset = datasets.STL10('./data',split='test',\
                                    transform=get_transform(config['ds_ft'],in_size),download=True)
        
    elif config['ds_ft']=='imagenet10':
        path_imagenet = save_dir/'data'/'imagenet10'/'val'
        testset = datasets.ImageFolder(path_imagenet,transform=get_transform(config['ds_ft'],in_size))
        
    elif config['ds_ft']=='tinyimagenet':
        path_tinyimagenet = save_dir/'data'/'tiny-imagenet-200'/'val'
        testset = datasets.ImageFolder(path_tinyimagenet,transform=get_transform(config['ds_ft'],in_size))
                                                                                  
    test_loader = DataLoader(testset,batch_size=100,num_workers=config['num_workers'],drop_last=False,shuffle=False)
  
    # defining loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # loading mcmc chekpoints
    ckt_list = os.listdir(os.path.join(exp_dir,'%s_%d'%(config['opt'],config['exp'])))
    n_ckts_tot = config['ckt_sp'] if config['ckt_sp'] else len(ckt_list) 
    
    n_ckts = n_ckts_tot - config['burn_in']
         
    for train_split in config['exp_split']:     
        print("\n--------------------")
        print("\nfine tunning on : {} % of {}".format(train_split,config['ds_ft']))
        print("--------------------\n")
        
        with open(os.path.join(split_dir,'train_set_%s_%d_%d%%.pickle'%(config['ds_ft'],in_size,train_split)),'rb') as f:
            train_set = pickle.load(f)

        with open(os.path.join(split_dir,'val_set_%s_%d_%d%%.pickle'%(config['ds_ft'],in_size,train_split)),'rb') as f:
            val_set = pickle.load(f) 
                                                                
        train_loader=DataLoader(train_set,batch_size=config['batch_size'],num_workers=config['num_workers'],\
                                drop_last=False,shuffle=True)
        val_loader=DataLoader(val_set,batch_size=config['batch_size'],num_workers=config['num_workers'],\
                              drop_last=False,shuffle=True)
        
        pr_tot = []
        log_tot= []    
        error_cycles = []  
        acc_cycles = []
        
        if config['fine']:
            for i in range(config['burn_in'],n_ckts_tot):
                print("\n ##################################################")
                print(f'{i+1}th model is loaded!')
                print("#####################################################")
            
                sam_dir = os.path.join(exp_dir ,'%s_%d'%(config['opt'],config['exp']))
                state_dict = torch.load(glob.glob(os.path.join(sam_dir,f'*_{i}.pt'))[0],map_location=device)
                if args.model_depth == 'res18':
                    backbone = models.resnet18(pretrained=False, progress=True)
                    fv_size = 512    # the size of embedding vector in resnet18 after avg pooling layer
                    
                elif args.model_depth == 'res50':
                    backbone = models.resnet50(pretrained=False, progress=True)
                    fv_size = 2048  # the size of embedding vector in resnet50 after avg pooling layer
                    
                if config['cont']=='byol':
                    net = networkbyol('online',backbone,config['mlp_hid_size'],config['proj_size']).to(device)
                    net.load_state_dict(state_dict)
                    encoder = nn.Sequential(*list(net.children())[:-2])
                    
                elif config['cont']=='simclr':
                    net = networksimclr(backbone,config['mlp_hid_size'],config['proj_size']).to(device)
                    net.load_state_dict(state_dict)
                    encoder = nn.Sequential(*list(net.children())[:-1])
                        
                model = finetune_net(encoder,fv_size,config['num_classes']).to(device)
                optimizer = optim.SGD(model.parameters(),lr=config['lr'],\
                                      nesterov=config['nes'],weight_decay=config['wd'],momentum=0.9)
                
                last = True if i==n_ckts_tot-1 else False
                save_dir_fine = save_dir / f'{args.cont}_ckpts' / 'finetune' / config['ds_ft']
                single_finetune(config['exp'],model,optimizer,criterion,train_loader,val_loader,\
                                config['num_epochs'],save_dir_fine,config['ckt'],i,device,train_split,\
                                config['ds_ft'],last,config['opt'])
                             
                print(f'\n making inference on test set for {i}th checkpoint!')
                ckt_inf = torch.load(os.path.join(save_dir_fine,'%s_%d%%_%s_%d_%s_model.pt'%
                                                  (config['exp'],train_split,config['ckt'],i,\
                                                   config['ckt_inf'])),map_location=device)
                                                                                                                                 
                model.load_state_dict(ckt_inf['model'])
                best_epoch = ckt_inf['epoch']                                               
                out_pr, logits, gt_list, error, acc = inference(model,test_loader,criterion,device)
                error_cycles.append(error)
                acc_cycles.append(acc)
                pr_tot.append(out_pr)
                log_tot.append(logits)
            
            pr_tot = np.stack(pr_tot,axis=0)
            log_tot = np.stack(log_tot,axis=0)
        
            error_cycles = np.array(error_cycles)
            acc_cycles = np.array(acc_cycles)
        
            # save gt
            np.save(os.path.join(result_dir_exp,f'gts.npy'),gt_list)
        
        # saving the results 
        if config['pr_ens']:
            np.save(os.path.join(result_dir_exp,'burn_%d_pr_ens.npy'%(config['burn_in'])),pr_tot)
            np.save(os.path.join(result_dir_exp,'burn_%d_logits_ens.npy'%(config['burn_in'])),log_tot)
            np.save(os.path.join(result_dir_exp,'burn_%d_error_cycl.npy'%(config['burn_in'])),error_cycles)
            np.save(os.path.join(result_dir_exp,'burn_%d_acc_cycl.npy'%(config['burn_in'])),acc_cycles)
        
        # evaluating on test set
        if config['eval']:
            if not config['pr_ens']:
                pr_tot = np.load(os.path.join(result_dir_exp,'burn_%d_pr_ens.npy'%(config['burn_in'])))
                log_tot = np.load(os.path.join(result_dir_exp,'burn_%d_logits_ens.npy'%(config['burn_in'])))
                gt_list = np.load(os.path.join(result_dir_exp,f'gts.npy'))
            nll_tot,acc_tot = evaluation_metrics(pr_tot,log_tot,gt_list,device,ig_sam=config['ig_sam'])
        if config['write_exp']:
            data = {
                'exp': [config['exp']],
                'opt':[config['opt']],
                'ds' :[config['ds_ft']],
                'ds_pr' :[config['ds_pr']],
                'split':[config['exp_split'][0]],
                'lr' :[config['lr']],
                'wd' :[config['wd']],
                'b_size':[config['batch_size']],
                'bur':[config['burn_in']],
                'ig_sam':[config['ig_sam']],
                'n_ensembel': [n_ckts],
                'NLL':[round(nll_tot.item(),4)],
                'ACC':[round(acc_tot,4)]}
            # ./results
            csv_path = result_dir_exp/'run_sweeps_test.csv'
            if os.path.exists(csv_path):
                sweeps_df = pd.read_csv(csv_path)
                sweeps_df = sweeps_df.append(
                pd.DataFrame.from_dict(data), ignore_index=True).set_index('exp')
            else:
                csv_path.parent.mkdir(parents=True, exist_ok=True) 
                sweeps_df = pd.DataFrame.from_dict(data).set_index('exp')
            # save experiment metadata csv file
            sweeps_df.to_csv(csv_path)
    print(f'\n Fine tunning was finished!\n')
    
##### Running finte tunning
if __name__=='__main__':
    semi_supervised(args)
         


















