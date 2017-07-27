import cv2
import numpy as np

from os import listdir

from openslide import OpenSlide

file_path_ground_truth_xml_17 = \
"/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training/lesion_annotations/"

save_location_path_ground_truth_jpg_17 = \
"/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_Ground_Truth_lv_4/tumor_mask_17"


def make_mask(mask_shpae, contours):

    wsi_empty = np.zeros(mask_shape[:2])
    wsi_empty = wsi_empty.astype(np.uint8)
    cv2.drawContours(wsi_empty, contours, -1, 255, -1)

    return wsi_empty





















