import torch
from torch.utils.data import Dataset, DataLoader, Subset

#### Data augmentation for pretraining
def get_transform(in_size,ds,s=1,aug=True):
    if ds == 'cifar10':
        mean_ds = [0.491, 0.482, 0.447]
        std_ds = [0.247, 0.243, 0.261] 
        
    elif ds == 'cifar100':
        mean_ds = [0.507, 0.486, 0.440]
        std_ds = [0.267, 0.256, 0.276] 
        
    elif ds == 'imagenet10' or ds == 'tinyimagenet':
        mean_ds = [0.485, 0.456, 0.406]
        std_ds = [0.229, 0.224, 0.225]
        
    elif ds == 'stl10':
        mean_ds = [0.485, 0.456, 0.406]
        std_ds = [0.229, 0.224, 0.225]
            
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    
    if aug:
        transform=transforms.Compose([transforms.RandomResizedCrop(size=in_size),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomApply([color_jitter], p=0.8),
                                  transforms.RandomGrayscale(p=0.2),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean_ds, std_ds)])
    else: 
        transform=transforms.Compose([transforms.Resize(size=(in_size,in_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean_ds, std_ds)])     
    return(transform)

class pair_aug(object):
    def __init__(self,transform):
        self.transform = transform        
    def __call__(self,img):
        if self.transform:
            img1= self.transform(img)
            img2= self.transform(img)
        return(img1,img2)
    
#### Dataloaders
def get_train_val_loader(dataset,val_size,batch_size,num_workers,seed):
    
    if val_size:
        split = ShuffleSplit(n_splits=1,test_size=val_size,random_state=seed)
        for train_idx, val_idx in split.split(range(len(dataset))):
            train_index= train_idx
            val_index = val_idx
            
        train_set = Subset(dataset,train_index)
        val_set = Subset(dataset,val_index)
        
        train_loader = DataLoader(train_set, batch_size=batch_size,num_workers=num_workers, drop_last=True, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size,num_workers=num_workers, drop_last=True, shuffle=True)
        return(train_loader,val_loader)
    
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,num_workers=num_workers, drop_last=True, shuffle=True)
        return(dataloader,_)
    
#### Preprocessing for finetune
def get_transformfinetune(ds,in_size=None):
    
    if ds == 'cifar10':
        mean_ds = [0.491, 0.482, 0.447]
        std_ds = [0.247, 0.243, 0.261]
        
    elif ds == 'cifar100':
        mean_ds = [0.507, 0.486, 0.440]
        std_ds = [0.267, 0.256, 0.276] 
        
    elif ds == 'stl10':
        mean_ds = [0.485, 0.456, 0.406]
        std_ds = [0.229, 0.224, 0.225]
        
    elif ds == 'imagenet10':
        mean_ds = [0.485, 0.456, 0.406]
        std_ds = [0.229, 0.224, 0.225]
            
    transform=transforms.Compose([transforms.Resize(size=(in_size,in_size)),transforms.ToTensor()])                                  
    return(transform)

