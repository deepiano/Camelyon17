import os
import cv2
import numpy as np
import math
import pdb
import csv

from os import listdir
from os.path import join

from skimage.filters import threshold_otsu
from skimage.transform.integral import integral_image, integrate
from openslide import OpenSlide
from openslide import open_slide
from matplotlib import pyplot as plt
from xml.etree.ElementTree import parse


### File path -Camelyon16

file_path_tif_of_tumor_slide_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Train_Tumor"
file_path_tif_of_normal_slide_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Train_Normal"
file_path_xml_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Ground_Truth/XML"
file_path_tis_msk_of_tumor_slide_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_16_Tumor"
file_path_tis_msk_of_normal_slide_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_16_Normal"
file_path_jpg_of_tumor_slide_16  = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_lv_4/Train_16_Tumor"
file_path_jpg_of_normal_slide_16  = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_lv_4/Train_16_Normal"
save_location_path_patch_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Train_patch_input"


def find_contours_of_xml_label(file_path_xml, downsample):

    list_blob = []
    tree = parse(file_path_xml)
    for parent in tree.getiterator():
        for index1, child1 in enumerate(parent):
            for index2, child2 in enumerate(child1):
               for index3, child3 in enumerate(child2):
                    list_point = []
                    for index4, child4 in enumerate(child3):
                        p_x = float(child4.attrib['X'])
                        p_y = float(child4.attrib['Y'])
                        p_x = p_x / downsample 
                        p_y = p_y / downsample 
                        list_point.append([p_x, p_y])
                    if len(list_point):
                        list_blob.append(list_point)

    contours = []
    for list_point in list_blob:
        list_point_int = [[[int(round(point[0])), int(round(point[1]))]] \
                            for point in list_point]
        contour = np.array(list_point_int, dtype=np.int32)
        contours.append(contour)

    return contours 


def get_list_file_name(path_directory):

    file_name_list = [name for name in listdir(path_directory)]
    file_name_list.sort()

    return file_name_list


def extract_patch(
            file_path_tif, \
            file_path_xml, \
            file_path_tis_mask, \
            file_path_jpg, \
            save_location_path_patch_position_visualize,\
            save_location_path_patch_position_csv,\
            size_patch, is_tumor_slide): 
    """
    -- Intput :

    file_path_tif : full path
    file_path_xml : full path
    file_path_tis_mask : full path
    file_path_jpg : full path
    save_location_path_patch_position_visualize : full path
    save_location_path_patch_position_csv : full path 

    -- Result :

    Draw patch position.
    Save coordinate of patch at level_0.

    """
    patch_level     = 1
    contours_level  = 4
    mask_level      = 4
    
    slide = OpenSlide(file_path_tif)
    slide_w_lv_4, slide_h_lv_4 = slide.level_dimensions[4]
    downsample = slide.level_downsamples[4]
    size_patch_lv_4 = int(size_patch / downsample)

    # Make integral image of slide
    tissue_mask = cv2.imread(file_path_tis_mask, 0)

    integral_image_tissue = integral_image(tissue_mask.T / 255)  

    # Load original bgr_jpg_lv_4 for visualizing patch position
    wsi_bgr_jpg = cv2.imread(file_path_jpg) 
    wsi_jpg_visualizing_patch_position = wsi_bgr_jpg.copy()

    print('==> making contours of tissue or tumor region from jpg or xml ..')

    # If Tumor_Slide, tumor regions exist.
    if is_tumor_slide == True:
 
        # Find and Draw contours_tumor - (color : yellow)
        contours_tumor = find_contours_of_xml_label(file_path_xml, downsample)
        cv2.drawContours(wsi_jpg_visualizing_patch_position, \
                        contours_tumor, -1, (0, 255, 255), 2)


    # Find and Draw contours_tissue - (color : blue)
    _, contours_tissue, _ = cv2.findContours( \
                                tissue_mask, \
                                cv2.RETR_TREE, \
                                cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(wsi_jpg_visualizing_patch_position, \
                    contours_tissue, -1, (255, 0, 0), 2)
    
    # Make csv_writer
    csv_file = open(save_location_path_patch_position_csv, 'wt')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow( ('X', 'Y') )

    print('==> Extracting patches randomly on tissue region...')
    patch_cnt = 0

    ### Extract random patches on tissue region
    for contour in contours_tissue:
        
        # Check if contour area is samller than patch area
        area = cv2.contourArea(contour)
        area_patch_lv_4 = size_patch_lv_4 ** 2
        if area < area_patch_lv_4:
            continue
        
        # Determine number of patches to extract
        number_patches = int(round(area / area_patch_lv_4 * 20))
        #number_patches = min(10000, number_patches)
#        print('contour area : ', area, ' num_patch : ', number_patches) 

        # Get coordinates of contour (level : 4)
        coordinates = (np.squeeze(contour)).T
        coords_x = coordinates[0]
        coords_y = coordinates[1]
        
        # Bounding box vertex 
        p_x_left = np.min(coords_x)
        p_x_right = np.max(coords_x)
        p_y_top = np.min(coords_y)
        p_y_bottom = np.max(coords_y)
        
        # Make candidates of patch coordinate (level : 4)
        candidate_x =\
                np.arange(round(p_x_left), round(p_x_right)).astype(int)
        candidate_y =\
                np.arange(round(p_y_top), round(p_y_bottom)).astype(int)

        
        # Pick coordinates randomly
        len_x = candidate_x.shape[0]
        len_y = candidate_y.shape[0]

        number_patches = min(number_patches, len_x)
        number_patches = min(number_patches, len_y)

        random_index_x = np.random.choice(len_x, number_patches, replace=False)
        random_index_y = np.random.choice(len_y, number_patches, replace=True)

        for i in range(number_patches):

            patch_x = candidate_x[random_index_x[i]] 
            patch_y = candidate_y[random_index_y[i]] 

            # Check if out of range
            if (patch_x + size_patch_lv_4 > slide_w_lv_4) or\
                (patch_y + size_patch_lv_4 > slide_h_lv_4) :
                
                continue

            # Check ratio of tumor region
            tissue_integral = integrate(integral_image_tissue,\
                                    (patch_x, patch_y),\
                                    (patch_x + size_patch_lv_4 - 1,
                                     patch_y + size_patch_lv_4 - 1))
            tissue_ratio = tissue_integral / (size_patch_lv_4 ** 2)

            if tissue_ratio < 0.8:
                continue

            # Save patches position to csv file. 
            patch_x_lv_0 = int(round(patch_x * downsample))
            patch_y_lv_0 = int(round(patch_y * downsample))
            csv_writer.writerow((patch_x_lv_0, patch_y_lv_0))
            patch_cnt += 1

            # Draw patch position (color : Green)
            cv2.rectangle(wsi_jpg_visualizing_patch_position, \
                            (patch_x, patch_y), \
                            (patch_x + size_patch_lv_4, patch_y + size_patch_lv_4), \
                            (0, 255, 0),\
                            thickness=1)

    print('patch_cnt: ', patch_cnt)

    # Save visualizing image.
    cv2.imwrite(save_location_path_patch_position_visualize,\
                 wsi_jpg_visualizing_patch_position)

    csv_file.close()


def extract_patch_on_slide(\
        file_path_tif,\
        file_path_xml,\
        file_path_tis_msk,\
        file_path_jpg,\
        save_location_path_patch,\
        is_tumor_slide):

    size_patch = 960
    
    file_name_list_tif = get_list_file_name(file_path_tif)
    file_name_list_xml = get_list_file_name(file_path_xml)
    file_name_list_tis_msk = get_list_file_name(file_path_tis_msk)
    file_name_list_jpg = get_list_file_name(file_path_jpg)
    
    for index in range(len(file_name_list_tif)):

        cur_path_tif = os.path.join(file_path_tif, file_name_list_tif[index])
        cur_path_xml = os.path.join(file_path_xml, file_name_list_xml[index])
        cur_path_tis_msk = os.path.join(file_path_tis_msk, 
                                        file_name_list_tis_msk[index])
        cur_path_jpg = os.path.join(file_path_jpg, file_name_list_jpg[index])

        cur_slide_name = file_name_list_tif[index].split('.')[0]
        cur_save_dir = os.path.join(\
                save_location_path_patch, cur_slide_name)

        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
        
        cur_jpg_file_name = cur_slide_name + '.jpg'
        cur_save_location_path_patch_pos_visualize =\
                os.path.join(cur_save_dir,\
                             cur_jpg_file_name)

        cur_csv_file_name = cur_slide_name + '.csv'
        cur_save_location_path_patch_pos_csv =\
                os.path.join(cur_save_dir,\
                            cur_csv_file_name )

        extract_patch(  
                cur_path_tif, \
                cur_path_xml, \
                cur_path_tis_msk, \
                cur_path_jpg, \
                cur_save_location_path_patch_pos_visualize,\
                cur_save_location_path_patch_pos_csv,\
                size_patch, \
                is_tumor_slide)



def main():

    extract_patch_on_slide(\
        file_path_tif_of_tumor_slide_16,\
        file_path_xml_16,\
        file_path_tis_msk_of_tumor_slide_16,\
        file_path_jpg_of_tumor_slide_16,\
        save_location_path_patch_16,
        is_tumor_slide=True)
    extract_patch_on_slide(\
        file_path_tif_of_normal_slide_16,\
        file_path_xml_16,\
        file_path_tis_msk_of_normal_slide_16,\
        file_path_jpg_of_normal_slide_16,\
        save_location_path_patch_16,
        is_tumor_slide=False)

if __name__=="__main__":
    main()
