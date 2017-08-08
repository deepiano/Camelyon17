import pdb
import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch

from torch.utils.data import Dataset

class FCNDataset(Dataset):

    class_names = np.array(['0', '1'])
    mean_bgr = np.array([0, 0, 0])

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        #dataset_dir = osp.join(self.root, )
        dataset_dir = self.root
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:

            imgsets_file = osp.join(dataset_dir, '%s.txt' % split)

            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, '%s.png' % did)
                lbl_file = osp.join(dataset_dir, '%s_lbl.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                    })


    def __len__(self):
        return len(self.files[self.split])


    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        img = img[:, :, 0:3] # RGBA -> RGB
        # load label
        lbl_file = data_file['lbl']
        print(lbl_file)
        lbl = PIL.Image.open(lbl_file)
        pdb.set_trace()
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1 # Why?
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl


    def transform(self, img, lbl):
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        pdb.set_trace()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    def untransform(self, img, lbl):
        img = img.numpy()
        pdb.set_trace()
        print(img)
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1] # BGR -> RGB
        lbl = lbl.numpy()
        return img, lbl
















