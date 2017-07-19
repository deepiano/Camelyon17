import cv2
import numpy as np
import math

from skimage.filters import threshold_otsu
from openslide import OpenSlide
from matplotlib import pyplot as plt
from xml.etree.ElementTree import parse

def get_level_scale_from_file_path_tif(file_path, level):

    wsi_tif = OpenSlide(file_path)
    level_scale_width = \
            wsi_tif.level_dimensions[level][0] / float(wsi_tif.dimensions[0])
    level_scale_height = \
            wsi_tif.level_dimensions[level][1] / float(wsi_tif.dimensions[1]) 
    level_scale = (level_scale_width, level_scale_height)

    return level_scale


def load_WSI_from_tif_file_path(file_path, level):
    
    wsi_tif = OpenSlide(file_path)
    wsi_pil = wsi_tif.read_region(\
                    (0, 0), level, wsi_tif.level_dimensions[level]) 
    return wsi_pil


def convert_wsi_pil_to_wsi_bgr(wsi_pil):
    
    wsi_array = np.array(wsi_pil)
    wsi_bgr = cv2.cvtColor(wsi_array, cv2.COLOR_RGBA2BGR)
    
    return wsi_bgr


def convert_wsi_bgr_to_wsi_gray(wsi_bgr):

    wsi_gray = cv2.cvtColor(wsi_bgr, cv2.COLOR_RGB2GRAY)
    
    return wsi_gray


def get_otsu_thresholding_image(wsi_gray):

    # opencv
    blur = cv2.GaussianBlur(wsi_gray, (5, 5), 0)
    ret, wsi_bin_uint8 = cv2.threshold(blur, 0, 255, \
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # scikit
    """
    threshold_global_otsu = threshold_otsu(wsi_gray)
    wsi_bool = wsi_gray <= threshold_global_otsu
    wsi_bin = wsi_bool.astype(np.uint8) 
    """

    return wsi_bin_uint8


def tissue_region_segmentation(file_path, level):

    wsi_pil = load_WSI_from_tif_file_path(file_path, level)
    wsi_bgr = convert_wsi_pil_to_wsi_bgr(wsi_pil)
    wsi_gray = convert_wsi_bgr_to_wsi_gray(wsi_bgr)
    wsi_bin_uint8 = get_otsu_thresholding_image(wsi_gray)

    return wsi_bin_uint8


def find_and_draw_contours_of_tissue_region(wsi_bin):
    
    wsi_with_contours, contours, hierarchy = \
            cv2.findContours(wsi_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    wsi_with_contours = cv2.cvtColor(wsi_with_contours, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(wsi_with_contours, contours, -1, (0, 255, 0), 3)

    return wsi_with_contours, contours 


def find_contours_of_xml_label(file_path_xml, level_scale):

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
                        p_x = p_x * level_scale[0]
                        p_y = p_y * level_scale[1]
                        list_point.append([p_x, p_y])
                    if len(list_point):
                        list_blob.append(list_point)

    contours = []
    for list_point in list_blob:
        list_point_int = [[[int(round(point[0])), int(round(point[1]))]] \
                            for point in list_point]
        contour = np.array(list_point_int, dtype=np.int32)
        contours.append(contour)

    #contours = [np.array([[[4000, 4000]], [[6000, 6000]], [[4000, 6000]]], dtype=np.int32),
    #            np.array([[[2000, 2000]],[[2000, 5000,]],[[5000, 5000]]], dtype=np.int32)]


    return contours 
                    

def draw_contours_of_label_on_wsi(wsi_bin, contours_label):
    pass


     

def process():
    # Tissue region segmentation.
    #   1. Load WSI from tif file path.
    #   2. Convert tif to opencv image.
    #   3. Otsu's thresholding.
    #   4. Save the complete image.
    #   => Problem : need more accuracy thresholding.
    # Extract contours of tissue region.
    #
    # (1)O contour -> bounding box
    # (2)X contour -> fill contours -> all coordinates
    #   1. np.nonzero
    #   2. np.ndarray.flatten
    #   3.
    # Extract patches on the contours. 
    #   0. Consider image_thresholded : At 40x (level 0),
    #                           patch : At 20x (level 1)
    #   1. At 20x 
    #   2. Size 960 x 960
    #   3. Pick a point randomly in the contours.
    #   4. Make the point the center of patch. 
    #   5. Extract patch and save the coordinate in txt.
    # Make a label responsive to each patch. 
    #     
    pass


def find_max_contour_in_contours(contours):

    max_contour_index = [-1, -1, -1, -1, -1]

    max_contour_index[0] = -1
    max_contour_size = -1
    for c_index, contour in enumerate(contours):
        size = contour.shape[0]
        if size > max_contour_size:
            max_contour_size = size
            max_contour_index[0] = c_index

    max_contour_size = -1
    for c_index, contour in enumerate(contours):
        size = contour.shape[0]
        if size > max_contour_size and c_index != max_contour_index[0] :
            max_contour_size = size
            max_contour_index[1] = c_index

    return contours[max_contour_index[1]]

def test():
    file_path = "./data/patient_099_node_4.tif"
    file_path_xml = "./data/patient_099_node_4.xml"
    level = 4
    level_scale = get_level_scale_from_file_path_tif(file_path, level)

    wsi_pil = load_WSI_from_tif_file_path(file_path, level)
    wsi_bgr = convert_wsi_pil_to_wsi_bgr(wsi_pil)
    wsi_gray = convert_wsi_bgr_to_wsi_gray(wsi_bgr)
    wsi_bin_uint8 = get_otsu_thresholding_image(wsi_gray)
    _, contours = find_and_draw_contours_of_tissue_region(wsi_bin_uint8) 
    wsi_with_contours = wsi_bgr.copy() 

    cv2.drawContours(wsi_with_contours, contours, -1, (0,255,0), 3)

    max_contour = find_max_contour_in_contours(contours) 

    wsi_with_max_contour = wsi_bgr.copy()
    cv2.drawContours(wsi_with_max_contour, [max_contour], 0, (0,255,0), -1)

    wsi_with_bounding_box = wsi_bgr.copy()
    #rect = cv2.minAreaRect(max_contour)
    #box = cv2.boxPoints(rect)
    #box = np.int32(box)
    #cv2.drawContours(wsi_with_bounding_box, [box], 0, (0, 0, 255), 2)
    x, y, w, h = cv2.boundingRect(max_contour)
    cv2.rectangle(wsi_with_bounding_box, (x,y), (x+w, y+h), (0, 0, 255), 2)


    contours_label = find_contours_of_xml_label(file_path_xml, level_scale)
     
    wsi_with_label = wsi_bgr.copy()
    cv2.drawContours(wsi_with_label, contours_label, 0, (0,255,0), -1)

    x, y, w, h = cv2.boundingRect(contours_label[0])
    cv2.rectangle(wsi_with_label, (x,y), (x+w, y+h), (0,0,255), 2)

    plt.subplot(1, 2, 1), plt.imshow(wsi_bgr)
    plt.title("wsi_bgr"), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2), plt.imshow(wsi_with_label)
    plt.title("wsi_label"), plt.xticks([]), plt.yticks([])

    plt.show()

    #cv2.imshow("wsi_bin", wsi_bin)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

if __name__=="__main__":
    test()
