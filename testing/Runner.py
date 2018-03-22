import os

import testing.params as pms
from commons.segmentation import AtureTest

base_dir = 'C:\\Projects\\ature'
os.chdir(base_dir)

data_file_path = 'data\\DRIVE\\test\\images'
mask_path = 'data\\DRIVE\\test\\mask'
ground_truth_path = 'data\\DRIVE\\test\\1st_manual'


def get_mask_file(file_name):
    return file_name.split('_')[0] + '_test_mask.gif'


def get_ground_truth_file(file_name):
    return file_name.split('_')[0] + '_manual1.gif'

params = {'sk_threshold': 60,
          'alpha': 7.5,
          'orig_contrib': 0.3,
          'seg_threshold': 24}

tester = AtureTest(out_dir='data\\drive_segmented_out')
tester.run_all(data_dir=data_file_path, mask_path=mask_path, gt_path=ground_truth_path, save_images=False,
               params_combination=[params], fget_mask=get_mask_file, fget_gt=get_ground_truth_file)
