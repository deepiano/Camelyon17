import cv2
import numpy as np
import math
import pdb
import os
import os.path

from os import listdir 
from os.path import isfile,join
from subprocess import call
#from skimage.filters import threshold_otsu
#from skimage.transform.integral import integral_image
from openslide import OpenSlide
#from openslide import open_slide
from matplotlib import pyplot as plt
#from xml.etree.ElementTree import parse


def find_max_width_height(li_im):
	w_max, h_max = -1, -1
	for im in li_im:
		shap = im.shape
		h, w = shap[0], shap[1]
		if w > w_max:
			w_max = w
		if h > h_max:
			h_max = h
	return w_max, h_max


def compute_rows_and_cols_4_1D_2_2D(n_img, w_img, h_img, th_r):
	cols_best, rows_best = -1, -1
	for rows in range(n_img, 0, -1):
		cols = int(math.ceil(float(n_img) / float(rows)))
		h_total = rows * h_img
		w_total = cols * w_img
		r_w_over_h = float(w_total) / float(h_total)
		if r_w_over_h >= th_r:
			cols_best = cols
			break
	if cols_best < 0:
		cols_best = n_img
		rows_best = 1
	else:
		rows_best = int(math.ceil(float(n_img) / float(cols_best)))
	return rows_best, cols_best


def make_merged_image_of_multiple_images(li_im, th_r):
	n_img = len(li_im)
	w_img, h_img = find_max_width_height(li_im)
	rows, cols = compute_rows_and_cols_4_1D_2_2D(n_img, w_img, h_img, th_r)
	n_channel = len(li_im[0].shape)
	if n_channel > 2:
		shap = (h_img * rows, w_img * cols) + li_im[0].shape[2:]
	else:
		shap = (h_img * rows, w_img * cols)
	im_merged = np.zeros(shap, np.uint8)
	i = 0
	shall_stop = False
	for r in range(rows):
		y_from = r * h_img
                y_to = y_from + h_img
		for c in range(cols):
			x_from = c * w_img
			x_to = x_from + w_img
			im_merged[y_from: y_to, x_from: x_to] = li_im[i]
			i += 1
			if i >= n_img:
				shall_stop = True
				break
		if shall_stop:
			break
	return im_merged


def generate_tiled_tif(fn_img, postfix=None):

    print('==> generating tif of ' + fn_img)
    #fn_tif = marked_image_path.replace('.untiled.tif', '.mark.tif')
    if None == postfix:
        fn_tif = os.path.splitext(fn_img)[0] + '.tif'
    else:
        fn_tif = os.path.splitext(fn_img)[0] + '_' + postfix + '.tif'
    
    cmd = 'vips tiffsave "%s" "%s" --compression=jpeg --vips-progress --tile --pyramid --tile-width=240 --tile-height=240' % (fn_img, fn_tif)
    #print(cmd)
    call(cmd, shell=True)


def run(file_path_origin, file_path_tismsk, save_location_path):

    print('==> saving merge slide at ' + save_location_path)
    im_bgr_origin = cv2.imread(file_path_origin) 
    im_bgr_result = cv2.imread(file_path_tismsk) 

#    im_rgb_origin = cv2.cvtColor(im_bgr_origin, cv2.COLOR_BGR2RGB)
#    im_rgb_result = cv2.cvtColor(im_bgr_result, cv2.COLOR_BGR2RGB) 

    im_bgr_merged = make_merged_image_of_multiple_images( \
                    [im_bgr_origin, im_bgr_result], 1.)

    cv2.imwrite(save_location_path, im_bgr_merged)
#    generate_tiled_tif(im_rgb_merged, save_location_path, postfix=merge)


if __name__=='__main__':

    file_path_origin_lv_4 = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_lv_4/Train_16_Tumor/"
    file_path_mask_lv_4 = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_16_Tumor/"
    
    file_path_origin_normal_lv_4 = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_lv_4/Train_16_Normal/"
    file_path_mask_normal_lv_4 = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_16_Normal/"

    file_path_origin_lv_7 = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_lv_7/"
    file_path_mask_lv_7 = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_7/"
    save_location_path_merge = \
    "/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_merge_orig_and_tismsk_lv_4/" 

### Merge tumor origin and tissue mask lv_4 and Save
    """
    list_origin_file_name = [f for f in listdir(file_path_origin_lv_4)]
    list_msk_file_name = [f for f in listdir(file_path_mask_lv_4)]
    list_origin_file_name.sort()
    list_msk_file_name.sort()

    for i in range(len(list_origin_file_name)):
        file_name_origin = list_origin_file_name[i] 
        file_name_mask = list_msk_file_name[i]
#        file_name_origin = slide_file_path_origin.split('/')[-1]
        file_name = file_name_origin.replace('_origin_lv_4.jpg', '')
        file_name = file_name + '_merge_lv_4.jpg' 
        cur_save_loca = save_location_path_merge + file_name 
        slide_file_path_origin = file_path_origin_lv_4 + file_name_origin
        slide_file_path_mask = file_path_mask_lv_4 + file_name_mask
        run(slide_file_path_origin, slide_file_path_mask, cur_save_loca)
    exit()
    """

    
### Merge tumor origin and tissue mask lv_4 and Save
    """ 
    list_merge_file_name = [f for f in listdir(save_location_path_merge)]
    list_merge_file_name.sort()

    for i in range(len(list_merge_file_name)):
       
        fn_img = save_location_path_merge + list_merge_file_name[i] 
        generate_tiled_tif(fn_img, postfix=None)
        if i==0: break
    """




### Merge Normal origin and tissue mask lv_4 and Save
    """
    list_origin_file_name = [f for f in listdir(file_path_origin_normal_lv_4)]
    list_msk_file_name = [f for f in listdir(file_path_mask_normal_lv_4)]
    list_origin_file_name.sort()
    list_msk_file_name.sort()

    for i in range(len(list_origin_file_name)):
        file_name_origin = list_origin_file_name[i] 
        file_name_mask = list_msk_file_name[i]
        file_name = file_name_origin.replace('_origin_lv_4.jpg', '')
        file_name = file_name + '_merge_lv_4.jpg' 
        cur_save_loca = save_location_path_merge + file_name 
        slide_file_path_origin = file_path_origin_normal_lv_4 + file_name_origin
        slide_file_path_mask = file_path_mask_normal_lv_4 + file_name_mask
        run(slide_file_path_origin, slide_file_path_mask, cur_save_loca)
#        if i == 0 : break
    exit()
    """

    
### Merge Normal origin and tissue mask lv_4 and Save

    list_merge_file_name = [f for f in listdir(save_location_path_merge)]
    list_merge_file_name.sort()

    for i in range(len(list_merge_file_name)):
        first_word = list_merge_file_name[i].split('_')[0]
        if first_word != 'normal': break
        fn_img = save_location_path_merge + list_merge_file_name[i]
        generate_tiled_tif(fn_img, postfix=None)
#        if i==0: break

### pipe tar across an ssh session shell script 
#$ tar czf - <files> | ssh user@host "cd /wherever && tar xvzf -"

















