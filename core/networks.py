import torch
from torch import nn
from torch import optim
import torch.nn.functional as F



import warnings 
warnings.filterwarnings('ignore')


#### Linear model
class MLP(nn.Module): 
    def __init__(self,in_dim,mlp_hid_size,proj_size):
        super(MLP,self).__init__()
        self.head = nn.Sequential(nn.Linear(in_dim,mlp_hid_size),
                                 nn.BatchNorm1d(mlp_hid_size),
                                 nn.ReLU(),
                                 nn.Linear(mlp_hid_size,proj_size))
        
    def forward(self,x):
        x= self.head(x)
        return(x)
    

#### Network for Byol model
class networkbyol(nn.Module):
    
    def __init__(self,net,backbone,mid_dim,out_dim):  
        super(network,self).__init__()
        
        self.net = net
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.projection = MLP(in_dim= backbone.fc.in_features,mlp_hid_size=mid_dim,proj_size=out_dim) 
        self.prediction = MLP(in_dim= out_dim,mlp_hid_size=mid_dim,proj_size=out_dim)
        
    def forward(self,x):
        
        embedding = self.encoder(x)
        embedding = embedding.view(embedding.size()[0],-1)
        project = self.projection(embedding)
        
        if self.net=='target':
            return(project)
        
        predict = self.prediction(project)
        return(predict)
    
    
    
#### Network for simclr model    
class networksimclr(nn.Module):
    
    def __init__(self,backbone,mid_dim,out_dim):  
        super(network,self).__init__()
        
        # we get representations from avg_pooling layer
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.projection = MLP(in_dim= backbone.fc.in_features,mlp_hid_size=mid_dim,proj_size=out_dim) 

        
    def forward(self,x):
        
        embedding = self.encoder(x)
        embedding = embedding.view(embedding.size()[0],-1)
        project = self.projection(embedding)
        
        return(project)
    
    
#### Building finetune model
class finetune_net(nn.Module):
    
    def __init__(self,model,in_dim,num_classes=0):
        super(finetune_net,self).__init__()
        
        self.model = model
        self.linear = nn.Linear(in_dim,num_classes)
        
    def forward(self,x):
        
        embeding = self.model(x)
        embeding = embeding.view(embeding.size()[0],-1)
        logits = self.linear(embeding) 
        
        return(logits)
    
    