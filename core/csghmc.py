import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer, required

import warnings 
warnings.filterwarnings('ignore')


DEFAULT_DAMPENING = 0.0
class SGHM(Optimizer): 
    def __init__(self,params,
                 lr=required,
                 momentum=0.99, 
                 dampening=0.,
                 weight_decay=0.,
                 N_train =0.,
                 temp= 1.0,
                 addnoise=True,
                 epoch_noise=True):    
        if weight_decay <0.0:
            raise ValueError("Invalid weight_decay value:{}".format(weight_decay))
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid leraning rate:{}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))   
        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay = weight_decay,
                        N_train = N_train,
                        temp = temp,
                        addnoise=addnoise,
                        epoch_noise=epoch_noise)
        super(SGHM, self).__init__(params, defaults)
    def step(self,closure=None):    
        """a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            N_train = group['N_train']
            temp = group['temp']
            epoch_noise= group['epoch_noise']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay!=0:
                    d_p.add_(p, alpha= weight_decay)
                d_p.mul_(-(1/2)* group['lr'])
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach() 
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum*(group['lr']/N_train)**0.5).add_(d_p, alpha=1 - dampening)
                    d_p = buf
                if group['addnoise'] and group['epoch_noise']:
                    noise = torch.randn_like(p.data).mul_((temp * group['lr']*(1-momentum)/N_train)**0.5)
                    p.data.add_(d_p +noise)
                    if torch.isnan(p.data).any(): exit('Nan param')
                    if torch.isinf(p.data).any(): exit('inf param')
                else:
                    p.data.add_(d_p)        
        return(loss)
