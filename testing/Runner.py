#!/home/akhanal1/Spring2018/pl-env/bin/python3.5
import os
import sys
import path_config as pth
import numpy as np
import itertools as itr
from random import shuffle
from testing.segmentation_test import AtureTestMat, AtureTest

sep = os.sep
data_file_path = pth.DATA_PATH + sep + 'DRIVE' + sep + 'test' + sep + 'images'
av_data = pth.DATA_PATH + sep + 'av_wide_data_set'

mask_path = pth.DATA_PATH + sep + 'DRIVE' + sep + 'test' + sep + 'mask'
ground_truth_path = pth.DATA_PATH + sep + 'DRIVE' + sep + 'test' + sep + '1st_manual'

# for ubuntu
sys.path.append(pth.CONTEXT_PATH)
os.chdir(pth.CONTEXT_PATH)


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

# Run for image files with in-time mask erosion
tester = AtureTest(data_dir=data_file_path, log_dir=os.path.join(pth.OUT_PATH, 'out_ak'))
tester.load_mask(mask_dir=mask_path, fget_mask_file=get_mask_file, erode_mask=True)
tester.load_ground_truth(ground_truth_dir=ground_truth_path, fget_ground_truth_file=get_ground_truth_file)
tester.run_for_one_image(file_name='01_test.tif', params_combination=[params], save=True)
tester.run_for_all_images(params_combination=all_params)

mask_path = pth.DATA_PATH + sep + 'DRIVE' + sep + 'test' + sep + 'mask_fixed'
# Run for mask fixed by Dr. Estrada
tester = AtureTest(data_dir=data_file_path, log_dir=os.path.join(pth.OUT_PATH, 'out_rj'))
tester.load_mask(mask_dir=mask_path, fget_mask_file=get_mask_file, erode_mask=False)
tester.load_ground_truth(ground_truth_dir=ground_truth_path, fget_ground_truth_file=get_ground_truth_file)
tester.run_for_one_image(file_name='01_test.tif', params_combination=[params], save=True)
tester.run_for_all_images(params_combination=all_params)

# Run for mat files in av_wide_data_set dir
tester = AtureTestMat(data_dir=data_file_path, log_dir=os.path.join(pth.OUT_PATH, 'out_ak'))
tester.load_mask(mask_dir=mask_path, fget_mask_file=get_mask_file)
tester.load_ground_truth(ground_truth_dir=ground_truth_path, fget_ground_truth_file=get_ground_truth_file)
tester.run_for_one_image(test_file_name='wide_image_03.mat', params=params)
