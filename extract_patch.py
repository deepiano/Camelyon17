import os
import cv2
import numpy as np
import math
import pdb

from os import listdir
from os.path import join

from skimage.filters import threshold_otsu
from skimage.transform.integral import integral_image, integrate
from openslide import OpenSlide
from openslide import open_slide
from matplotlib import pyplot as plt
from xml.etree.ElementTree import parse


### File path -Camelyon16

file_path_tif_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Train_Tumor"
file_path_xml_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Ground_Truth/XML"
file_path_tumor_msk_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_Ground_Truth_lv_4/tumor_mask_16"
file_path_tis_msk_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_tissue_mask_lv_4/Train_16_Tumor"
file_path_jpg_16  = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_origin_lv_4/Train_16_Tumor"
save_location_path_tumor_patch_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Train_patch_input/Input_16_Tumor"
save_location_path_normal_patch_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Train_patch_input/Input_16_Normal"

test_path = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/test/test.jpg"

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


def visualize_one(img):

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img", 1000, 1000)
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_list_file_name(path_directory):

    file_name_list = [name for name in listdir(path_directory)]
    file_name_list.sort()

    return file_name_list

def extract_patch(
            file_path_tif, \
            file_path_xml, \
            file_path_tumor_mask, \
            file_path_tis_mask, \
            file_path_jpg, \
            save_location_path_tumor_patch, \
            save_location_path_normal_patch, \
            size_patch, is_tumor_slide): 
    """

    file_path_tif : full path
    file_path_xml : full path
    file_path_tumor_mask : full path
    file_path_tis_mask : full path
    file_path_jpg : full path
    save_location_path_tumor_patch : full path
    save_location_path_normal_patch : full path

    """
    patch_level     = 1
    contours_level  = 4
    mask_level      = 4
    

    slide = OpenSlide(file_path_tif)
    downsample = slide.level_downsamples[4]
    size_patch_lv_4 = int(round(size_patch / downsample))

    # Make integral image of slide
    tumor_mask = cv2.imread(file_path_tumor_mask, 0) 
    tissue_mask = cv2.imread(file_path_tis_mask, 0)

    integral_image_tumor = integral_image(tumor_mask.T / 255)
    integral_image_tissue = integral_image(tissue_mask.T / 255)  


    ### If Tumor_Slide

    if is_tumor_slide == True:
 
        print('==> making contours of tumor region from xml ..')
 
        contours_tumor = find_contours_of_xml_label(file_path_xml, downsample)
#        _, contours_tissue, _ = cv2.findContours( \
#                                    tissue_mask, \
#                                    cv2.RETR_TREE, \
#                                    cv2.CHAIN_APPROX_SIMPLE) 

        wsi_bgr_jpg = cv2.imread(file_path_jpg) 
        wsi_jpg_visualizing_patch_position = wsi_bgr_jpg.copy()

        ### Draw contours_tumor and contours_tissue
        cv2.drawContours(wsi_jpg_visualizing_patch_position, \
                        contours_tumor, -1, (0, 255, 255), 3)
#        cv2.drawContours(wsi_jpg_visualizing_patch_position, \
#                        contours_tissue, -1, (255, 0, 0), 3)
    
        ### Extract Tumor patches
        for contour in contours_tumor:
            
            # Check if contour area is samller than patch area
            area = cv2.contourArea(contour)
            area_patch = size_patch * size_patch
            if area < area_patch:
                continue
            
            # Determine number of patches to extract
            number_patches = int(round(area / area_patch * 500))
            number_patches = min(10000, number_patches)

            # Get coordinates of contour
            coordinates = (np.squeeze(contour)).T
            coords_x = coordinates[0]
            coords_y = coordinates[1]
            
            # Bounding box vertex 
            p_x_left = np.min(coords_x)
            p_x_right = np.max(coords_x)
            p_y_top = np.min(coords_y)
            p_y_bottom = np.max(coords_y)
            
#            print('integral shape : ', integral_image_tumor.shape)
#            print('area : ', area)
#            print('number_patches : ', number_patches)
#            print('x_left : ', p_x_left)
#            print('x_right : ', p_x_right)
#            print('y_top : ', p_y_top)
#            print('y_bottom : ', p_y_bottom)
            
            # Make candidates of patch coordinate
            candidate_x =\
                    np.arange(round(p_x_left), round(p_x_right)).astype(int)
            candidate_y =\
                    np.arange(round(p_y_top), round(p_y_bottom)).astype(int)
            
            # Pick coordinates 
            len_x = candidate_x.shape[0]
            len_y = candidate_y.shape[0]

            number_patches = min(number_patches, len_x)
            number_patches = min(number_patches, len_y)

            random_index_x = np.random.choice(len_x, number_patches, replace=False)
            random_index_y = np.random.choice(len_y, number_patches, replace=True)
            
            list_patch_coordinate = []

            for i in range(number_patches):
                patch_x = candidate_x[random_index_x[i]] 
                patch_y = candidate_y[random_index_y[i]] 
                list_patch_coordinate.append(np.array([patch_x, patch_y]))

                # Check ratio of tumor region
                tissue_integral = integrate(integral_image_tumor,\
                                        (patch_x, patch_y),\
                                        (patch_x + size_patch_lv_4 - 1,
                                            patch_y + size_patch_lv_4 - 1))
                tissue_ratio = tissue_integral / (size_patch_lv_4 ** 2)
                print ('x, y : ', patch_x, patch_y)       
                print('tis_integral : ', tissue_integral)
                print('tis ratio :', tissue_ratio)
                if tissue_ratio < 0.8:
                    continue
                print('pick!')

                # Draw patch position 
                cv2.rectangle(wsi_jpg_visualizing_patch_position, \
                                (patch_x, patch_y), \
                                (patch_x + size_patch_lv_4, patch_y + size_patch_lv_4), \
                                (0, 255, 255),\
                                thickness=1)

        cv2.imwrite(test_path, wsi_jpg_visualizing_patch_position)
        exit()
      


#            break
        cv2.imwrite(test_path, wsi_jpg_visualizing_patch_position)
        exit()



### Extract Normal patches
     
    coords_y4, coords_x4 = np.nonzero(tissue_mask_lv_4)
    length_coord = coord_x4.shape[0]
    random_index = np.random.choice(length_coord, 10, replace=False)
    list_patch_location = []

    list_title = []
    list_patch_img = []

    downsample_lv_4 = slide.level_downsamples[4]
    coord_x0 = np.round(coord_x4 * downsample_lv_4)
    coord_y0 = np.round(coord_y4 * downsample_lv_4)
    coord_x0 = coord_x0.astype(np.int32)
    coord_y0 = coord_y0.astype(np.int32)

    size_patch_lv_0 = size_patch
    size_patch_lv_4 = int(round(size_patch / downsample_lv_4)) 

    check_patch_choice_ary = np.zeros(wsi_gray_lv_4.shape, dtype=np.uint8)
    slide_w, slide_h = slide.dimensions
   
    print('length : ', length_coord)
    pdb.set_trace()

    index = -1000
    cnt = 0

######## start while
    while len(list_patch_location) <= 10:
        index += 1000
        x0 = coord_x0[index]
        y0 = coord_y0[index]

# check if out of range
        if (x0 + size_patch_lv_0) > slide_w or \
           (y0 + size_patch_lv_0) > slide_h:
            continue

        x4 = int(round(x0 / downsample_lv_4))
        y4 = int(round(y0 / downsample_lv_4))

# check if already chosen
        if check_patch_choice_ary[y4][x4] == 255:
            continue

# check ratio of tissue region 
        mask_patch = tissue_mask_lv_4[y4:y4+size_patch_lv_4, \
                                      x4:x4+size_patch_lv_4]

        mask_patch = mask_patch / 255
        print(np.max(mask_patch))
        print(mask_patch.shape)

        tissue_ratio = \
                np.sum(mask_patch) / (float)(size_patch_lv_4*size_patch_lv_4)
        print(tissue_ratio)

        if tissue_ratio < 0.8:
            continue

# pick the patch
        check_patch_choice_ary[y4][x4] = 255
        patch_location = np.array([x0, y0])
        list_patch_location.append(patch_location) 

# visualizing patch
        patch_pil = slide.read_region( \
                (x0, y0), 0, (size_patch_lv_0,size_patch_lv_0))
        patch_ary = np.array(patch_pil)
        patch_bgr = cv2.cvtColor(patch_ary, cv2.COLOR_RGBA2BGR) 
        location_path = "./visual/patch_" + str(cnt+1) + ".png"
        cv2.imwrite(location_path, patch_bgr) 
        list_title.append('patch ' + str(cnt))
        list_patch_img.append(patch_bgr)

        cnt += 1

# visualizing (x0, y0) -> (x4, y4) dot on wsi_lv_4
        x4 = int(round(x0 / downsample_lv_4))
        y4 = int(round(y0 / downsample_lv_4))

        cv2.rectangle(wsi_bgr_lv_4_circle, \
                (x4, y4), \
                (x4+size_patch_lv_4,y4+size_patch_lv_4),\
                (0,255,0), thickness=5)
######### finish while
    
    print('==> drawing patch..')

    cv2.imwrite("./visual/draw_dot.png", wsi_bgr_lv_4_circle)

    for i in range(10):
        plt.subplot(2, 5, i+1), plt.imshow(list_patch_img[i])
        plt.title(list_title[i]), plt.xticks([]), plt.yticks([])

    plt.show()



def get_integral_image(mask_0to1):
    pass
    


    """
def extract_patch_from_label_mask_at_lv_1 \
        (file_path_tif, file_path_xml, file_path_jpg_lv_4, \
            save_location_path, \
            size_patch, is_tumor_slide): 

    patch_level     = 1
    contours_level  = 4

    file_name_list_tif = [name for name in listdir(file_path_tif)]
    file_name_list_tif.sort()
    
    for file_name_tif in file_name_list_tif:

        slide = OpenSlide(file_path)
        downsample = slide.level_downsamples[4]

        if is_tumor_slide == True:
    
            print('==> making contours of tumor region from xml ..')
    
            file_name_list_xml = [name for name in listdir(file_path_xml)]
            file_name_list_xml.sort()
        
            for file_name in file_name_list_xml:
        
                cur_path_xml = file_path_xml + file_name
                contours = find_contours_of_xml_label(cur_path_xml, downsample)
        
                wsi_bgr_lv_4 = cv2.imread( 

    wsi_pil_lv_4 = slide.read_region((0, 0), 4,\
                             slide.level_dimensions[4])
    wsi_ary_lv_4 = np.array(wsi_pil_lv_4)
    wsi_bgr_lv_4 = cv2.cvtColor(wsi_ary_lv_4, cv2.COLOR_RGBA2BGR)
    wsi_gray_lv_4 = cv2.cvtColor(wsi_bgr_lv_4, cv2.COLOR_BGR2GRAY)

    blur_lv_4 = cv2.GaussianBlur(wsi_gray_lv_4, (5, 5), 0)
    ret, wsi_bin_0255_lv_4 = cv2.threshold(blur_lv_4, 0, 255, \
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    _, contours_tissue_lv_4, hierarchy = \
            cv2.findContours(wsi_bin_0255_lv_4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
     
    print('==> extracting patch ..')

    wsi_bgr_lv_4_visualizing_patch_position = wsi_bgr_lv_4.copy()
     
    coord_y4, coord_x4 = np.nonzero(tissue_mask_lv_4)
    length_coord = coord_x4.shape[0]
    random_index = np.random.choice(length_coord, 10, replace=False)
    list_patch_location = []

    list_title = []
    list_patch_img = []

    downsample_lv_4 = slide.level_downsamples[4]
    coord_x0 = np.round(coord_x4 * downsample_lv_4)
    coord_y0 = np.round(coord_y4 * downsample_lv_4)
    coord_x0 = coord_x0.astype(np.int32)
    coord_y0 = coord_y0.astype(np.int32)

    size_patch_lv_0 = size_patch
    size_patch_lv_4 = int(round(size_patch / downsample_lv_4)) 

    check_patch_choice_ary = np.zeros(wsi_gray_lv_4.shape, dtype=np.uint8)
    slide_w, slide_h = slide.dimensions
   
    print('length : ', length_coord)
    pdb.set_trace()

    index = -1000
    cnt = 0

######## start while
    while len(list_patch_location) <= 10:
        index += 1000
        x0 = coord_x0[index]
        y0 = coord_y0[index]

# check if out of range
        if (x0 + size_patch_lv_0) > slide_w or \
           (y0 + size_patch_lv_0) > slide_h:
            continue

        x4 = int(round(x0 / downsample_lv_4))
        y4 = int(round(y0 / downsample_lv_4))

# check if already chosen
        if check_patch_choice_ary[y4][x4] == 255:
            continue

# check ratio of tissue region 
        mask_patch = tissue_mask_lv_4[y4:y4+size_patch_lv_4, \
                                      x4:x4+size_patch_lv_4]

        mask_patch = mask_patch / 255
        print(np.max(mask_patch))
        print(mask_patch.shape)

        tissue_ratio = \
                np.sum(mask_patch) / (float)(size_patch_lv_4*size_patch_lv_4)
        print(tissue_ratio)

        if tissue_ratio < 0.8:
            continue

# pick the patch
        check_patch_choice_ary[y4][x4] = 255
        patch_location = np.array([x0, y0])
        list_patch_location.append(patch_location) 

# visualizing patch
        patch_pil = slide.read_region( \
                (x0, y0), 0, (size_patch_lv_0,size_patch_lv_0))
        patch_ary = np.array(patch_pil)
        patch_bgr = cv2.cvtColor(patch_ary, cv2.COLOR_RGBA2BGR) 
        location_path = "./visual/patch_" + str(cnt+1) + ".png"
        cv2.imwrite(location_path, patch_bgr) 
        list_title.append('patch ' + str(cnt))
        list_patch_img.append(patch_bgr)

        cnt += 1

# visualizing (x0, y0) -> (x4, y4) dot on wsi_lv_4
        x4 = int(round(x0 / downsample_lv_4))
        y4 = int(round(y0 / downsample_lv_4))

        cv2.rectangle(wsi_bgr_lv_4_circle, \
                (x4, y4), \
                (x4+size_patch_lv_4,y4+size_patch_lv_4),\
                (0,255,0), thickness=5)
######### finish while
    
    print('==> drawing patch..')

    cv2.imwrite("./visual/draw_dot.png", wsi_bgr_lv_4_circle)

    for i in range(10):
        plt.subplot(2, 5, i+1), plt.imshow(list_patch_img[i])
        plt.title(list_title[i]), plt.xticks([]), plt.yticks([])

    plt.show()
    """


###### Debugging draw dot 
"""
##########################
    coord_y4, coord_x4 = np.nonzero(wsi_label_mask_level_4)

    length_coord = coord_y4.shape[0]
    print ('=>drawing circles..')
    for i in range(length_coord):

        if i % 100 == 0:
            cv2.circle(wsi_bgr_lv_4, (coord_x4[i], coord_y4[i]), 3, (0,255,0), \
                        thickness=2)


    coord_y, coord_x = np.nonzero(wsi_label_mask_level_1)

    level_scale_1 = slide.level_downsamples[1]
    level_scale_4 = slide.level_downsamples[4]
    coord_y4d = coord_y * level_scale_1
    coord_x4d = coord_x * level_scale_1
    coord_y4d = np.round(coord_y4d / level_scale_4)
    coord_x4d = np.round(coord_x4d / level_scale_4)
    coord_y4d = coord_y4d.astype(np.int32)
    coord_x4d = coord_x4d.astype(np.int32)
    length_coord = coord_y4d.shape[0]

    wsi_bgr_lv_4d = wsi_bgr_lv_4.copy()

    print ('=>drawing circles 2..')
    for i in range(length_coord):

        if i % 1000 == 0:
            cv2.circle(wsi_bgr_lv_4d, (coord_x4d[i], coord_y4d[i]), 3, (0,255,0), \
                        thickness=2)
    
    print ('=> showing image..')

    plt.subplot(1, 2, 1), plt.imshow(wsi_bgr_lv_4)
    plt.title("level_4"), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2), plt.imshow(wsi_bgr_lv_4d)
    plt.title("level_4 debug"), plt.xticks([]), plt.yticks([])
    
    cv2.imwrite('./visiual/wsi_bgr_lv_4debug.png', wsi_bgr_lv_4d)
    plt.show()
#######################
    """


def main():

    size_patch = 960
    
    file_name_list_tif = get_list_file_name(file_path_tif_16)
    file_name_list_xml = get_list_file_name(file_path_xml_16)
    file_name_list_tumor_msk = get_list_file_name(file_path_tumor_msk_16)
    file_name_list_tis_msk = get_list_file_name(file_path_tis_msk_16)
    file_name_list_jpg = get_list_file_name(file_path_jpg_16)
    
    index = 8
    cur_path_tif = os.path.join(file_path_tif_16, file_name_list_tif[index])
    cur_path_xml = os.path.join(file_path_xml_16, file_name_list_xml[index])
    cur_path_tumor_msk = os.path.join(file_path_tumor_msk_16, 
                                    file_name_list_tumor_msk[index])
    cur_path_tis_msk = os.path.join(file_path_tis_msk_16, 
                                    file_name_list_tis_msk[index])
    cur_path_jpg = os.path.join(file_path_jpg_16, file_name_list_jpg[index])
    cur_save_location_path_tumor_patch = "" 
    cur_save_location_path_normal_patch = ""
    is_tumor_slide = True

    extract_patch(  
            cur_path_tif, \
            cur_path_xml, \
            cur_path_tumor_msk, \
            cur_path_tis_msk, \
            cur_path_jpg, \
            cur_save_location_path_tumor_patch, \
            cur_save_location_path_normal_patch, \
            size_patch, \
            is_tumor_slide)




if __name__=="__main__":
    main()
