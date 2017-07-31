import cv2
import numpy as np

from os import listdir
from xml.etree.ElementTree import parse

from openslide import OpenSlide


file_path_source_slide_17 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training/"

file_path_origin_jpg_17= \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_lv_4/Train_17_Tumor/"

file_path_ground_truth_xml_17 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training/lesion_annotations/"

save_location_path_ground_truth_jpg_17 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_Ground_Truth_lv_4/tumor_mask_17/"


def make_mask(mask_shape, contours):

    wsi_empty = np.zeros(mask_shape[:2])
    wsi_empty = wsi_empty.astype(np.uint8)
    cv2.drawContours(wsi_empty, contours, -1, 255, -1)

    return wsi_empty


def find_contours_of_xml(file_path_xml, downsample):

    list_blob = []
    tree = parse(file_path_xml)
    for parent in tree.getiterator():
        for index_1, child_1 in enumerate(parent):
            for index_2, child_2 in enumerate(child_1):
                for index_3, child_3 in enumerate(child_2):
                    list_point = []
                    for index_4, child_4 in enumerate(child_3):
                        p_x = float(child_4.attrib['X'])
                        p_y = float(child_4.attrib['Y'])
                        p_x = p_x / downsample 
                        p_y = p_y / downsample 
                        list_point.append([p_x, p_y])
                    if len(list_point) >= 0:
                        list_blob.append(list_point)

    contours = []
    for list_point in list_blob:
        list_point_int = [[[int(round(point[0])), int(round(point[1]))]] \
                            for point in list_point]
        contour = np.array(list_point_int, dtype=np.int32)
        contours.append(contour)

    return contours


def save_tumor_mask_jpg(file_path_origin, file_path_xml, \
                         save_location_path, downsample):

    file_name_list_origin = [name for name in listdir(file_path_origin)] 
    file_name_list_origin.sort()

    file_name_list_xml = [name for name in listdir(file_path_xml)] 
    file_name_list_xml.sort()
    len_origin = len(file_name_list_origin)
    len_xml = len(file_name_list_xml) 

    for i, file_name in enumerate(file_name_list_xml):
       
        print ('==> Finding contours of ' + file_name + ' ..') 
        cur_path_xml = file_path_xml + file_name
        contours = find_contours_of_xml(cur_path_xml, downsample)

        print ('==> Making mask of ' + file_name + ' ..') 
        cur_path_origin = file_path_origin + file_name_list_origin[i]
        wsi_bgr = cv2.imread(cur_path_origin)
        mask_shape = wsi_bgr.shape[0:2]
        tumor_mask_0255 = make_mask(mask_shape, contours) 
        
        print ('==> Saving maks at ' + save_location_path + ' ..') 
        file_name = file_name.replace('.xml', '')
        file_name = file_name + '_mask_lv_4.jpg' 
        cur_path_save = save_location_path + file_name
        cv2.imwrite(cur_path_save, tumor_mask_0255)


if __name__=='__main__':

    slide = OpenSlide(file_path_source_slide_17 + \
    'centre_4/patient_099_node_0.tif')
    downsample = slide.level_downsamples[4]
    save_tumor_mask_jpg(file_path_origin_jpg_17, \
                        file_path_ground_truth_xml_17, \
                        save_location_path_ground_truth_jpg_17, \
                        downsample) 
    
























