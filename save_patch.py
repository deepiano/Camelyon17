import os
from os import listdir
from os.path import join


import cv2
import numpy as np
from openslide import OpenSlide
import csv


# File path -Camelyon16
file_path_tif_of_tumor_slide_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Train_Tumor"
file_path_tif_of_normal_slide_16 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Train_Normal"

# File path -Camelyon17
file_path_tif_17 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training"

# Patch Save Location
save_location_path_patch = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Train_patch_input"

def main():

    patch_level = 1
    size_patch  = 960
    
    # Camelyon16 tumor slide
    save_patch(\
        file_path_tif_of_tumor_slide_16,
        save_location_path_patch,
        patch_level,
        size_patch)

    # Camelyon16 normal slide
    save_patch(\
        file_path_tif_of_normal_slide_16,
        save_location_path_patch,
        patch_level,
        size_patch)

#    # Camelyon17 slide
#    for i in range(5):
#        dir_name = 'centre_' + str(i)
#        cur_file_path_tif = os.path.join(file_path_tif_17, dir_name)  
#
#        save-patch(\
#                cur_file_path_tif,
#                save_location_path_patch,
#                patch_level,
#                size_patch)


def get_list_file_name(path_directory):

    file_name_list = [name for name in listdir(path_directory)]
    file_name_list.sort()

    return file_name_list

def save_patch(\
        file_path_tif,
        save_location_path_patch,
        patch_level,
        size_patch):
    
    print('=> Making patch ..')

    # Read file name list of slide directory
    file_name_list_tif = get_list_file_name(file_path_tif)

    for index in range(len(file_name_list_tif)):

        file_name_tif = file_name_list_tif[index]
        file_name_slide = file_name_tif.split('.')[0]
        file_name_csv = file_name_slide + '.csv'

        cur_file_path_tif =\
                os.path.join(file_path_tif, file_name_tif)
        cur_file_path_csv =\
                os.path.join(save_location_path_patch, file_name_slide)
        cur_file_path_csv =\
                os.path.join(cur_file_path_csv, file_name_csv)

        # Load slide image
        slide = OpenSlide(cur_file_path_tif)

        # Read patch position csv file
        csv_file = open(cur_file_path_csv, 'rb')
        csv_reader = csv.DictReader(csv_file)
        csv_data = list(csv_reader)

        # Make patch and Save 
        for i, row in enumerate(csv_data):

            x = int(row['X'])
            y = int(row['Y'])
            patch = slide.read_region((x, y),\
                                patch_level,\
                                (size_patch, size_patch))
            patch = np.array(patch)
            patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2BGR)
            
            file_name_patch_jpg = file_name_slide + '_patch_' + str(i) + '.jpg'
            cur_save_path_patch_jpg =\
                    os.path.join(save_location_path_patch,\
                                 file_name_slide)
            cur_save_path_patch_jpg =\
                    os.path.join(cur_save_path_patch_jpg ,\
                                 file_name_patch_jpg)
            cv2.imwrite(cur_save_path_patch_jpg, patch)
        
        csv_file.close()

        print('=> Next..')

if __name__=='__main__':
    main()
