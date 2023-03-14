import numpy as np
import torch
def mixup_data(x, y, use_mixup=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if use_mixup == True:
        lam = np.random.random()
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).cuda()

    xprem = x[index, :]
    
    mixed_x = lam * x + (1 - lam) * xprem
    y, yprem = y, y[index]
    return mixed_x, y, yprem, lam

def mixup_criterion(criterion, pred, y, yprem, lam):
    return lam * criterion(pred, y) + (1 - lam) * criterion(pred, yprem)