import testing.params as pms
from commons.segmentation import AtureTest
import os
base_dir = 'C:\\Projects\\ature'
os.chdir(base_dir)

data_file_path = 'data\\DRIVE\\test\\images'
mask_path = 'data\\DRIVE\\test\\mask'
ground_truth_path = 'data\\DRIVE\\test\\1st_manual'
mask_suffix = '_test_mask.gif'
ground_truth_suffix = '_manual1.gif'

tester = AtureTest(out_dir='out')
tester.run_all(data_dir=data_file_path, mask_path=mask_path, gt_path=ground_truth_path, save_images=False,
               params_combination=pms.get_param_combinations())
