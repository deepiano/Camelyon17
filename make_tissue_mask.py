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
#from xml.etree.ElementTree import parse


file_path_tumor = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Train_Tumor/" 
file_path_normal = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Train_Normal/" 
file_path_ground_truth_tif = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Ground_Truth/Mask/" 

save_location_path_origin_lv_4 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_lv_4/"
save_location_path_origin_normal_lv_4 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_lv_4/Train_16_Normal/"
save_location_path_mask_tumor_lv_4 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_16_Tumor/"
save_location_path_mask_normal_lv_4 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_16_Normal/"

save_location_path_mask_lv_7 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_7/"
save_location_path_cutting_lv_7 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_cutting_lv_7/"
save_location_path_cutting_lv_4 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_cutting_lv_4/"
save_location_path_ground = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_Ground_Truth_lv_4/"

### Camelyon17 path
file_path_slide_17 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training/"
file_path_normal_slide_17 = \
""
file_path_ground_truth_xml_17 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training/lesion_annotations/"

save_location_path_origin_tumor_17 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_lv_4/Train_17_Tumor/"
save_location_path_origin_normal_17 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_lv_4/Train_17_Normal/"
save_location_path_tumor_tissue_mask_17 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_17_Tumor/"
save_location_path_normal_tissue_mask_17 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_17_Normal/"


def make_mask(mask_shape, contours):

    wsi_empty = np.zeros(mask_shape[:2])
    wsi_empty = wsi_empty.astype(np.uint8)
    cv2.drawContours(wsi_empty, contours, -1, 255, -1)
    
    return wsi_empty 


def run(file_path, location_path, level, padding, camel_17):
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

    if camel_17:
        wsi_bgr_lv_black = (wsi_bgr_lv_ == 0)
        wsi_bgr_lv_[wsi_bgr_lv_black] = 255

### Remove black region.
    """
    wsi_bgr_lv_sum = np.sum(wsi_bgr_lv_, 2)
    wsi_criterion = (wsi_bgr_lv_sum / 3) < 38
    wsi_bgr_lv_[wsi_criterion] = np.array([255, 255, 255])
    """

    
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
    kernel_c = np.ones((4,4), dtype=np.uint8)
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

def is_tumor_slide(cur_file_name, list_file_name_xml):
    
    cur_file_name = cur_file_name.split('.')[0]
    
    for i, file_name_xml in enumerate(list_file_name_xml):
        file_name_xml = file_name_xml.split('.')[0]
        if cur_file_name == file_name_xml:
            return True 

    return False 



def save_origin_slide(file_path, save_location, list_file_name_xml, save_tumor):

    list_file_name = [f for f in listdir(file_path)]
    list_file_name.sort()
    level = 4
    for i, file_name in enumerate(list_file_name):
        
        if save_tumor:
            if (is_tumor_slide(file_name, list_file_name_xml) == False):
                continue
        else:
            if (is_tumor_slide(file_name, list_file_name_xml)):
                continue

        cur_file_path = file_path + file_name
        file_name = file_name.lower()
        file_name = file_name.replace('.tif', '')
        file_name = file_name + '_origin_lv_' + str(level) + '.jpg'
        # check if correct path
        cur_save_loca = save_location + file_name 
        save_slide_as_jpg_with_level(cur_file_path, cur_save_loca, level)
#        if i == 0: break


def save_tissue_mask(file_path, save_location, list_file_name_xml, save_tumor):

    list_file_name = [f for f in listdir(file_path)]
    list_file_name.sort()

    level = 4
    for i, file_name in enumerate(list_file_name):

        if save_tumor:
            if (is_tumor_slide(file_name, list_file_name_xml) == False):
                continue
        else:
            if (is_tumor_slide(file_name, list_file_name_xml) == True):
                continue

        cur_file_path = file_path+ file_name 
        file_name = file_name.lower()
        file_name = file_name.replace('.tif', '')
        file_name = file_name + '_tissue_mask_lv_' + str(level) + '.jpg'
        # check if correct path
        cur_save_loca = save_location + file_name 
        padding = False 
        camel_17 = True
        run(cur_file_path, cur_save_loca, level, padding, camel_17)
#        if i == 0: break


if __name__=='__main__':

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

### Camelyon 17

    list_file_name_xml = \
            [f for f in listdir(file_path_ground_truth_xml_17)] 

### Save normal origin slide bgr lv 4 -Camelyon17
    """
    for i in range(5):
        file_path = file_path_slide_17 + 'centre_' + str(i) +'/'
        save_origin_slide(file_path, save_location_path_origin_normal_17, \
                            list_file_name_xml, False)
    exit()
    """
    
### Save tumor -
    """
    for i in range(5):
        file_path = file_path_slide_17 + 'centre_' + str(i) +'/'
        save_origin_slide(file_path, save_location_path_origin_tumor_17, \
                            list_file_name_xml, True)
    exit()
    """

    """
### Save normal tissue mask -Camelyon17
    for i in range(5):
        file_path = file_path_slide_17 + 'centre_' + str(i) +'/'
        save_tissue_mask(file_path, save_location_path_normal_tissue_mask_17, \
                            list_file_name_xml, False)
    exit()
    """


### Save turmor -
    for i in range(5):
        file_path = file_path_slide_17 + 'centre_' + str(i) +'/'
        save_tissue_mask(file_path, save_location_path_tumor_tissue_mask_17, \
                            list_file_name_xml, True)
    exit()

## Save tumor slide tissue region mask lv_4 
    list_file_name = [f for f in listdir(file_path_tumor)]
    list_file_name.sort()

    level = 4
    for i, file_name in enumerate(list_file_name):
        cur_file_path = file_path_tumor + file_name 
        file_name = file_name.lower()
        file_name = file_name.replace('.tif', '')
        file_name = file_name + '_tissue_mask_lv_' + str(level) + '.jpg'
        # check if correct path
        cur_save_loca = save_location_path_mask_tumor_lv_4 + file_name 
        padding = True
        if i >= 70: padding = False 

        run(cur_file_path, cur_save_loca, level, padding)
#        if i == 0: break
    exit()



### Save normal slide tissue region mask lv_4 
    list_file_name = [f for f in listdir(file_path_normal)]
    list_file_name.sort()

    level = 4
    for i, file_name in enumerate(list_file_name):
        cur_file_path = file_path_normal + file_name 
        file_name = file_name.lower()
        file_name = file_name.replace('.tif', '')
        file_name = file_name + '_tissue_mask_lv_' + str(level) + '.jpg'
        # check if correct path
        cur_save_loca = save_location_path_mask_normal_lv_4 + file_name 

        padding = False 
        run(cur_file_path, cur_save_loca, level, padding)
#        if i == 0: break

























