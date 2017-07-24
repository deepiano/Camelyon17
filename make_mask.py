import cv2
import numpy as np
import math
import pdb
from os import listdir
from os.path import isfile,join

from skimage.filters import threshold_otsu
from skimage.transform.integral import integral_image
from openslide import OpenSlide
#from openslide import open_slide
from matplotlib import pyplot as plt
from xml.etree.ElementTree import parse


def make_mask(mask_shape, contours):

    wsi_empty = np.zeros(mask_shape[:2])
    wsi_empty = wsi_empty.astype(np.uint8)
    cv2.drawContours(wsi_empty, contours, -1, 255, -1)
    
    return wsi_empty 


def run(file_path, location_path, level, padding):

    slide = OpenSlide(file_path)
    
    print('==> making contours of tissue region..')

### Pad with 255 
    if padding == True:
        x_lv_, y_lv_ = 0, 0
        w_lv_, h_lv_ = slide.level_dimensions[level]
    
        wsi_pil_lv_ = slide.read_region((0,0), level,\
            (w_lv_, h_lv_))
    
        wsi_ary_lv_ = np.array(wsi_pil_lv_)
        wsi_bgr_lv_ = cv2.cvtColor(wsi_ary_lv_, cv2.COLOR_RGBA2BGR)
    
        margin_top = int(round(h_lv_ / 12.))
        margin_bottom = int(round(h_lv_ / 32.))
        wsi_bgr_lv_[0:margin_top, :] = 255
        wsi_bgr_lv_[h_lv_ - margin_bottom:h_lv_, :] = 255

    else:
        wsi_pil_lv_ = slide.read_region((0, 0), level,\
                slide.level_dimensions[level])
        wsi_ary_lv_ = np.array(wsi_pil_lv_)
        wsi_bgr_lv_ = cv2.cvtColor(wsi_ary_lv_, cv2.COLOR_RGBA2BGR)

### Remove black region.
    wsi_bgr_lv_sum = np.sum(wsi_bgr_lv_, 2)
    wsi_criterion = (wsi_bgr_lv_sum / 3) < 38
    wsi_bgr_lv_[wsi_criterion] = np.array([255, 255, 255])

    
### Visualizing
#    origin = wsi_bgr_lv_.copy()
#    plt.subplot(1, 2, 1), plt.imshow(origin)
#    plt.title("origin"), plt.xticks([]), plt.yticks([])
#    plt.subplot(1, 2, 2), plt.imshow(wsi_bgr_lv_4 )
#    plt.title("after"), plt.xticks([]), plt.yticks([])
#    plt.show()
#    exit() 

    wsi_gray_lv_ = cv2.cvtColor(wsi_bgr_lv_, cv2.COLOR_BGR2GRAY)

#    ret, wsi_bin_0255_lv_4 = cv2.threshold( \
#                          wsi_gray_lv_4, \
#                          127, 255, \
#                          cv2.THRESH_BINARY ) 

### Visualizing
#    plt.subplot(1, 2, 1), plt.imshow(wsi_bgr_lv_4 )
#    plt.title("bgr"), plt.xticks([]), plt.yticks([])
#    plt.subplot(1, 2, 2), plt.imshow(wsi_gray_lv_4, 'gray')
#    plt.title("gray"), plt.xticks([]), plt.yticks([])
#    plt.show()
#    exit()

#    blur_lv_ = cv2.GaussianBlur(wsi_gray_lv_, (5, 5), 0)
    ret, wsi_bin_0255_lv_ = cv2.threshold( \
                    wsi_gray_lv_, 0, 255, \
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

### Visualizing
#    plt.subplot(1, 2, 1), plt.imshow(wsi_bgr_lv_4 )
#    plt.title("bgr"), plt.xticks([]), plt.yticks([])
#    plt.subplot(1, 2, 2), plt.imshow(wsi_bin_0255_lv_4, 'gray')
#    plt.title("gray"), plt.xticks([]), plt.yticks([])
#    plt.show()
#    exit()

### Morphology
    
    kernel_o = np.ones((2,2), dtype=np.uint8)
    kernel_c = np.ones((5,5), dtype=np.uint8)
    wsi_bin_0255_lv_ = cv2.morphologyEx( \
            wsi_bin_0255_lv_, \
            cv2.MORPH_CLOSE, \
            kernel_c)
    wsi_bin_0255_lv_ = cv2.morphologyEx( \
            wsi_bin_0255_lv_, \
            cv2.MORPH_OPEN, \
            kernel_o)

    _, contours_tissue_lv_, hierarchy = \
            cv2.findContours(\
                    wsi_bin_0255_lv_, \
                    cv2.RETR_TREE, \
                    cv2.CHAIN_APPROX_SIMPLE)
     
    print('==> making tissue mask..')

    mask_shape_lv_ = wsi_gray_lv_.shape
    tissue_mask_lv_ = make_mask(mask_shape_lv_, contours_tissue_lv_) 

    print('==> saving slide_lv_' + str(level) + ' at ' + location_path)
    cv2.imwrite(location_path, tissue_mask_lv_)

### Visualizing one mask
#    plt.subplot(1, 1, 1), plt.imshow(tissue_mask_lv_, 'gray')
#    plt.xticks([]), plt.yticks([])
#    plt.show()

### Visualizing mask
#    plt.subplot(1, 2, 1), plt.imshow(tissue_mask_lv_4, 'gray')
#    plt.xticks([]), plt.yticks([])
#    plt.subplot(1, 2, 2), plt.imshow(wsi_bgr_lv_4)
#    plt.xticks([]), plt.yticks([])
#    plt.show()

def save_slide_as_jpg_with_level(file_path, save_location, level):
                
    slide_tif = OpenSlide(file_path) 
    print(('==> saving slide_lv_%s at ' + save_location) % level)
 
    wsi_pil_lv_ = slide_tif.read_region((0, 0), level,\
        slide_tif.level_dimensions[level])
    wsi_ary_lv_ = np.array(wsi_pil_lv_)
    wsi_bgr_lv_ = cv2.cvtColor(wsi_ary_lv_, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(save_location, wsi_bgr_lv_)
        
def save_slide_cutting(file_path, save_location, level):
                
    slide = OpenSlide(file_path) 
    print('==> saving slide_lv_' + str(level) + ' at ' + save_location)
    
    x_lv_, y_lv_ = 0, 0
    w_lv_, h_lv_ = slide.level_dimensions[level]

    wsi_pil_lv_ = slide.read_region((0,0), level,\
        (w_lv_, h_lv_))

    wsi_ary_lv_ = np.array(wsi_pil_lv_)
    wsi_bgr_lv_ = cv2.cvtColor(wsi_ary_lv_, cv2.COLOR_RGBA2BGR)

    margin_top = int(round(h_lv_ / 12.))
    margin_bottom = int(round(h_lv_ / 32.))
    wsi_bgr_lv_[0:margin_top, :] = 255
    wsi_bgr_lv_[h_lv_ - margin_bottom:h_lv_, :] = 255

    cv2.imwrite(save_location, wsi_bgr_lv_)

#    plt.subplot(1,1,1), plt.imshow(wsi_bgr_lv_)
#    plt.xticks([]), plt.yticks([])
#    plt.show()

if __name__=='__main__':

    file_path_tif = \
    "/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Train_Tumor/" 
    file_path_ground_truth_tif = \
    "/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Ground_Truth/Mask/" 
    save_location_path_mask_lv_4 = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_16_Tumor/"
    save_location_path_mask_lv_7 = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_7/"
    save_location_path_cutting_lv_7 = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_cutting_lv_7/"
    save_location_path_cutting_lv_4 = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_cutting_lv_4/"
    save_location_path_ground = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_Ground_Truth_lv_4/"

    list_file_name = [f for f in listdir(file_path_tif)]
    list_file_name.sort()
    
#    file_name = list_file_name[0]
#    cur_file_path = file_path_tif + file_name 
#    file_name = file_name.lower()
#    file_name = file_name.replace('.tif', '')
#    file_name = file_name + '_tissue_mask.jpg'
#    cur_save_loca = save_location_path + file_name 
#    print(cur_file_path)
#    print(cur_save_loca)
#    print('\n')
#    run(cur_file_path, cur_save_loca)
#    exit()

### Save origin slide with padding lv 4 (No.1 ~ 70)
    """
    level = 4
    for i, file_name in enumerate(list_file_name):
        if i == 70 : break
        cur_file_path = file_path_tif + file_name 
        file_name = file_name.lower()
        file_name = file_name.replace('.tif', '')
        file_name = file_name + '_origin_cut_lv_' + str(level) + '.jpg'
        cur_save_loca = save_location_path_cutting_lv_4 + file_name 
        save_slide_cutting(cur_file_path, cur_save_loca, level)
    exit()
    """


### Save gound truth lv 4 
    """ 
    list_file_name_ground = [f for f in listdir(file_path_ground_truth_tif)]
    list_file_name_ground.sort()
    level = 4
    for i, file_name in enumerate(list_file_name_ground):
        cur_file_path = file_path_ground_truth_tif + file_name
        file_name = file_name.lower()
        file_name = file_name.replace('.tif', '')
        file_name = file_name + '_lv_' + str(level) + '.jpg'
        # check if correct path
        cur_save_loca = save_location_path_ground + file_name 

        save_slide_as_jpg_with_level(cur_file_path, cur_save_loca, level)
#        if i == 0: break
    exit()
    """

### Save tissue region mask lv_4 
    level = 4
    for i, file_name in enumerate(list_file_name):
        cur_file_path = file_path_tif + file_name 
        file_name = file_name.lower()
        file_name = file_name.replace('.tif', '')
        file_name = file_name + '_tissue_mask_lv_' + str(level) + '.jpg'
        # check if correct path
        cur_save_loca = save_location_path_mask_lv_4 + file_name 
#        img = cv2.imread(cur_save_loca, 0)
#        if img is not None:
#            continue
#
#        print(cur_file_path)
#        print(cur_save_loca)
#        print('\n')
        padding = True
        if i >= 70: padding = False 

        run(cur_file_path, cur_save_loca, level, padding)
        if i == 1: break





























