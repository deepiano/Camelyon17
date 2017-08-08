import pdb
import os
import os.path as osp

import numpy as np
import torch
import fcn_datasets

def main():
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)

    if cuda:
        torch.cuda.manual_seed(1337)
 
    # 1. dataset
    root = osp.expanduser('~/TeamProject/Camelyon17/fcn')
    train_dataset = fcn_datasets.FCNDataset(root, split='train', transform=True)
 
    train_loader = torch.utils.data.DataLoader(train_dataset,\
          batch_size=1,\
          shuffle=False,)
 
    for i, (input, target) in enumerate(train_loader):
        print(i)
        print(input)
        print(target)
        

if __name__=='__main__':
   main()
