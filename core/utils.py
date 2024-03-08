import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path 


def plotCurves(stats,results_dir=None):
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1,1,1)
    plt.plot(stats['train'], label='train_loss')
    plt.plot(stats['val'], label='valid_loss')
    textsize = 12
    marker=5
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('NLL')
    lgd = plt.legend(['train', 'validation'], markerscale=marker, 
                 prop={'size': textsize, 'weight': 'normal'})    
    plt.savefig(results_dir , bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    
def plotCurvesFinetune(loss,acc,exp,ckt,split,i,ds,save_dir=None,opt='sghm'):
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.plot(loss['train'], label='train_loss')
    plt.plot(loss['val'], label='valid_loss')
    textsize = 12
    marker=5
    plt.xlabel('Epochs')
    plt.ylabel(f'Loss')
    plt.title(f'NLL on {split}% data')
    lgd = plt.legend(['train', 'validation'], markerscale=marker, 
                 prop={'size': textsize, 'weight': 'normal'}) 
    ax = plt.gca()
    plt.subplot(1,2,2)
    plt.plot(acc['train'], label='train')
    plt.plot(acc['val'], label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.title(f'Accuracy on {split}% data')
    lgd = plt.legend(['train', 'validation'], markerscale=marker, 
                 prop={'size': textsize, 'weight': 'normal'})
    crv_dir = save_dir / f'{ds}_exp_{exp}_opt_{opt}_fine_{ckt}_semi_{split}%_model_{i}_loss.png'
    plt.savefig(crv_dir, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    
    
def get_accuracy(gt, pred):
    assert len(gt)==len(pred)
    right = 0
    for i in range(len(gt)):
        if gt[i]==pred[i]:
             right += 1.0
    return right/len(gt)


def evaluation_metrics(pr_tot,logits,gt_list,device,ig_sam=0): 
    pr_tot = torch.from_numpy(pr_tot[ig_sam:])
    logits = torch.from_numpy(logits[ig_sam:])
    tot_ens = pr_tot.size()[0]
    mean_logits = torch.mean(logits,dim=0).to(device)
    mean_pr = torch.mean(pr_tot,dim=0).to(device)
    _, pred_label = torch.max(mean_pr, dim=1)
    pred_list = list(pred_label.data)
    acc = get_accuracy(gt_list, pred_list)
    gt = torch.tensor(gt_list).to(device)
    nll = nn.CrossEntropyLoss()(mean_logits,gt)
    print('\n')
    print(f'########## Total Accuracy and NLL for {tot_ens} Ens ###########')    
    print(f'\n total nll is: {nll:0.4f}, and total accuracy is: {acc:0.4f}')    
    print(f'#############################################')
    return(nll,acc)
    
    
