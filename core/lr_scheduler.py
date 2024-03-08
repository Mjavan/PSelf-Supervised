
import math
import torch



min_v = 0 

def update_lr(lr0,batch_idx,cycle_batch_length,n_sam_per_cycle,optimizer):
            
    is_end_of_cycle = False
        
    prop = batch_idx % cycle_batch_length
    
    pfriction = prop/cycle_batch_length
    
    lr = lr0 * (min_v +(1.0-min_v)*0.5*(np.cos(np.pi * pfriction)+1.0)) 
            
    if prop >= cycle_batch_length-n_sam_per_cycle:

        is_end_of_cycle = True

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr


def get_lr(optimizer):
    
    for group in optimizer.param_groups:
        
        return(group['lr'])

def update_lin(lr,optimizer):
    
    for param_group in optimizer.param_groups:

        param_group['lr'] = lr   
        
        
        