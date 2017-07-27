import cv2
import numpy as np

import os
from os import listdir
"""
We already have tissue region mask and tumor region mask of Tumor Slide.
This program makes Normal mask of Tumor Slide.
Method :
    tissue region mask - tumor region mask (subtract)
"""

### File Path -Camelyon16
path_tumor_slide_16_tissue_mask = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_16_Tumor/"
path_tumor_slide_16_tumor_mask = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_Ground_Truth_lv_4/tumor_mask_16/"
path_save_location_16_normal_mask = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_16_Normal_of_Tumor/"


### File Path -Camelyon16
path_tumor_slide_17_tissue_mask = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_16_Tumor/"
path_tumor_slide_17_tumor_mask = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_Ground_Truth_lv_4/tumor_mask_17/"
path_save_location_17_normal_mask = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_16_Normal_of_Tumor/"


def make_normal_mask(path_tis_msk, path_tumor_msk, path_save_location):

    print('==> making normal mask...')

    tis_msk = cv2.imread(path_tis_msk)
    tumor_msk = cv2.imread(path_tumor_msk)

    
    tumor_msk_bool = (tumor_msk == 255)
    tis_msk_after = tis_msk.copy()
    tis_msk_after[tumor_msk_bool] = 0

    print('==> saving normal mask at' + path_save_location + ' ...')
    cv2.imwrite(path_save_location, tis_msk_after)

### Display result
    """
    cv2.namedWindow('tis_msk', cv2.WINDOW_NORMAL)
    cv2.namedWindow('tis_msk_after', cv2.WINDOW_NORMAL)
    cv2.namedWindow('tumor_msk', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('tis_msk', 512, 512)
    cv2.resizeWindow('tis_msk_after', 512, 512)
    cv2.resizeWindow('tumor_msk', 512, 512)
    cv2.imshow('tis_msk', tis_msk)
    cv2.imshow('tis_msk_after', tis_msk_after)
    cv2.imshow('tumor_msk', tumor_msk)
    cv2.waitKey()
    cv2.destoryAllWindows()
    """

if __name__=='__main__':
    
    list_file_name_tissue_mask = [name for name in \
            listdir(path_tumor_slide_16_tissue_mask)]
    list_file_name_tissue_mask.sort()
    list_file_name_tumor_mask = [name for name in \
            listdir(path_tumor_slide_16_tumor_mask)]
    list_file_name_tumor_mask.sort()

    len_ = len(list_file_name_tissue_mask) 
    for i in range(len_):
        file_name_tissue_mask = list_file_name_tissue_mask[i]
        file_name_tumor_mask = list_file_name_tumor_mask[i]
        cur_path_tissue = path_tumor_slide_16_tissue_mask + \
                            file_name_tissue_mask
        cur_path_tumor = path_tumor_slide_16_tumor_mask + \
                            file_name_tumor_mask

        file_name_save_word = file_name_tissue_mask.split('_') 
        file_name_save = \
                file_name_save_word[0] + '_' + file_name_save_word[1] + \
                '_normal_mask_lv_4.jpg'
        cur_path_save = path_save_location_16_normal_mask + \
                        file_name_save
#        print(cur_path_tissue)
#        print(cur_path_tumor)
#        print(file_name_save)
        make_normal_mask(cur_path_tissue, cur_path_tumor, cur_path_save) 
#        if i == 0: break


        

