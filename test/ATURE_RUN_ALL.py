# coding: utf-8

# In[ ]:


import os

base_path = "C:\\Projects\\ature\\"  # WINDOWS
# base_path = "home/ak/Projects/ature/" # LINUX

data_file_path = base_path + "\\data\\DRIVE\\test\\images"
mask_path = base_path + "\\data\\DRIVE\\test\\mask"
ground_truth_path = base_path + "\\data\\DRIVE\\test\\1st_manual"
log_path = base_path + "\\log"

os.chdir(base_path)

from commons.IMAGE import Image
from commons.ImgLATTICE import Lattice
import preprocess.utils.img_utils as imgutils
from commons.MAT import Mat
from PIL import Image as IMG
import numpy as np
from commons import constants as const
import cv2
from preprocess.algorithms import fast_mst as fmst
import itertools as itr


# In[ ]:


#########Load av wide mat file#########
# os.chdir(pth.join(data_path, 'av_wide_data_set'))
# file = Mat(file_name='wide_image_06.mat')
# original = file.get_image('I2')
# img = Image(image_arr=original[:,:,1])
# img.apply_bilateral()
# img.apply_gabor(kernel_bank=imgutils.get_chosen_gabor_bank() )


# In[ ]:


#######Load image directly##########
# os.chdir(data_file_path)
# original = IMG.open('01_test.tif')
# original = np.array(original.getdata(), np.uint8).reshape(original.size[1], original.size[0], 3)
# img = Image(image_arr=original[:,:,1])
# img.apply_bilateral()
# img.apply_gabor(kernel_bank=imgutils.get_chosen_gabor_bank())


# In[ ]:


# -------CONSTANTS--------
# SKELETONIZE_THRESHOLD = 20

# # Image lattice constants
# IMG_LATTICE_COST_ASSIGNMENT_ALPHA = 5

# IMG_LATTICE_COST_GABOR_IMAGE_CONTRIBUTION = 0.6

# # MST algorithm parameters
# SEGMENTATION_THRESHOLD = 8


# In[ ]:


def run(img_obj, lattice_obj, params, mask, truth):
    ##### Unpack all params
    SKELETONIZE_THRESHOLD, IMG_LATTICE_COST_ASSIGNMENT_ALPHA, IMG_LATTICE_COST_GABOR_IMAGE_CONTRIBUTION, SEGMENTATION_THRESHOLD = params

    ##### Create skeleton based on threshold
    img_obj.create_skeleton(threshold=SKELETONIZE_THRESHOLD, kernels=imgutils.get_chosen_skeleton_filter())
    seed_node_list = imgutils.get_seed_node_list(img_obj.img_skeleton)

    ##### Run segmnetation
    graph = fmst.run_segmentation(image_object=img_obj,
                                  lattice_object=lattice_obj,
                                  seed_list=seed_node_list,
                                  segmentation_threshold=SEGMENTATION_THRESHOLD,
                                  alpha=IMG_LATTICE_COST_ASSIGNMENT_ALPHA,
                                  img_gabor_contribution=IMG_LATTICE_COST_GABOR_IMAGE_CONTRIBUTION,
                                  img_original_contribution=1 - IMG_LATTICE_COST_GABOR_IMAGE_CONTRIBUTION)

    ##### Apply mask
    segmented = cv2.bitwise_and(lattice_obj.accumulator, lattice_obj.accumulator, mask=mask)

    ##### Calculate F1 measure
    TP = 0  # True Positive
    FP = 0  # False Positive
    FN = 0  # False Negative
    for i in range(0, segmented.shape[0]):
        for j in range(0, segmented.shape[1]):
            if segmented[i, j] == 255 and truth[i, j] == 255:
                TP += 1
            if segmented[i, j] == 255 and truth[i, j] == 0:
                FP += 1
            if segmented[i, j] == 0 and truth[i, j] == 0:
                FN += 1
    F_score = 2 * TP / (2 * TP + FP + FN)

    ##### Log result
    parms = str(round(F_score, 3)) + ',' + str(SKELETONIZE_THRESHOLD) + ',' + str(
        IMG_LATTICE_COST_ASSIGNMENT_ALPHA) + ',' + str(IMG_LATTICE_COST_GABOR_IMAGE_CONTRIBUTION) + ',' + str(
        SEGMENTATION_THRESHOLD)
    log_file.write(parms + '\n')
    log_file.flush()


# In[ ]:


############# ENTRY POINT HERE ###############
############################################
SK_THRESHOLD_PARAMS = np.arange(0, 50, 10)
ALPHA_PARAMS = np.arange(5, 11, 1)
GABOR_CONTRIBUTION_PARAMS = np.arange(0.5, 1.1, 0.1)
SEGMENTATION_THRESHOLD_PARAMS = np.arange(6, 22, 2)

PARAMS_COMBINATION = itr.product(SK_THRESHOLD_PARAMS, ALPHA_PARAMS, GABOR_CONTRIBUTION_PARAMS,
                                 SEGMENTATION_THRESHOLD_PARAMS)

#### Work on all images in a directory
os.chdir(data_file_path)
for test_image in os.listdir(os.getcwd()):

    print('WORKING ON: ' + test_image)
    original = IMG.open(test_image)
    original = np.array(original.getdata(), np.uint8).reshape(original.size[1], original.size[0], 3)
    img_obj = Image(image_arr=original[:, :, 1])
    img_obj.apply_bilateral()
    img_obj.apply_gabor(kernel_bank=imgutils.get_chosen_gabor_bank())

    #### Load the corresponding mask as array
    os.chdir(mask_path)
    #### Read image as array
    mask_file = test_image[:2] + '_test_mask.gif'
    mask = IMG.open(mask_file)
    print("MASK LOADED: " + mask_file)
    mask = np.array(mask.getdata(), np.uint8).reshape(mask.size[1], mask.size[0], 1)[:, :, 0]

    #### Load ground truth segmented result as an array
    os.chdir(ground_truth_path)
    ground_truth_file = test_image[:2] + '_manual1.gif'
    truth = IMG.open(ground_truth_file)
    print("GROUND TRUTH LOADED: " + ground_truth_file)
    truth = np.array(truth.getdata(), np.uint8).reshape(truth.size[1], truth.size[0], 1)[:, :, 0]

    lattice_obj = Lattice(image_arr_2d=img_obj.img_gabor)
    lattice_obj.generate_lattice_graph()

    os.chdir(log_path)
    log_file = open(test_image + "_log.csv", 'w')
    log_file.write(
        "FSCORE,SKELETONIZE_THRESHOLD,IMG_LATTICE_COST_ASSIGNMENT_ALPHA,IMG_LATTICE_COST_GABOR_IMAGE_CONTRIBUTION,SEGMENTATION_THRESHOLD\n")

    for params in PARAMS_COMBINATION:
        run(img_obj, lattice_obj, params, mask, truth)
    log_file.close()

