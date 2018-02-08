import os

### CHANGE path HERE
base_path_win = "C:\\Projects\\ature\\"  # WINDOWS
base_path_lin = "/home/akhanal1/Spring2018/ature"  # LINUX

if os.name == 'nt':
    base_path = base_path_win
else:
    base_path = base_path_lin

sep = os.sep
data_file_path = base_path + sep + 'data' + sep + 'DRIVE' + sep + 'test' + sep + 'images'
# data_file_path = base_path + sep + 'data' + sep + 'av_wide_data_set'

mask_path = base_path + sep + 'data' + sep + 'DRIVE' + sep + 'test' + sep + 'mask'
ground_truth_path = base_path + sep + 'data' + sep + 'DRIVE' + sep + 'test' + sep + '1st_manual'

os.chdir(base_path)

import numpy as np
import itertools as itr
from random import shuffle
from testing.segmentation_test import AtureTestErode, AtureTestMat


def get_mask_file(file_name): return file_name.split('_')[0] + '_test_mask.gif'


def get_ground_truth_file(file_name): return file_name.split('_')[0] + '_manual1.gif'


SK_THRESHOLD_PARAMS = np.arange(40, 61, 20)
ALPHA_PARAMS = np.arange(5, 7, 0.5)
GABOR_CONTRIBUTION_PARAMS = np.arange(0.9, 1.4, 0.2)
SEGMENTATION_THRESHOLD_PARAMS = np.arange(8, 15, 0.5)

PARAMS_ITR = itr.product(SK_THRESHOLD_PARAMS, ALPHA_PARAMS, GABOR_CONTRIBUTION_PARAMS, SEGMENTATION_THRESHOLD_PARAMS)

PARAMS_COMBINATION = list(PARAMS_ITR)
shuffle(PARAMS_COMBINATION)

keys = ('sk_threshold', 'alpha', 'gabor_contrib', 'seg_threshold')

all_params = (dict(zip(keys, param)) for param in PARAMS_COMBINATION)

params = {'sk_threshold': 60,
          'alpha': 5.0,
          'gabor_contrib': 1.3,
          'seg_threshold': 10.5}

############# Run for images in data dir ###############
############################################
tester = AtureTestErode(data_path=data_file_path)
tester.load_mask(mask_path=mask_path, fget_mask_file=get_mask_file)
tester.load_ground_truth(ground_truth_path=ground_truth_path, fget_ground_truth_file=get_ground_truth_file)
tester.run_for_one_image(test_file_name='02_test.tif', params_combination=all_params)
# tester.run_for_all_images(params=params)


############# Run for mat files in av_wide_data_set dir ###############
############################################
tester = AtureTestMat(data_path=data_file_path)
tester.load_mask(mask_path=mask_path, fget_mask_file=get_mask_file)
tester.load_ground_truth(ground_truth_path=ground_truth_path, fget_ground_truth_file=get_ground_truth_file)
# tester.run_for_one_image(test_file_name='wide_image_03.mat', params=params)
