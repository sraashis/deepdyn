import os
import sys

sys.path.append('/home/akhanal1/ature')
os.chdir('/home/akhanal1/ature')

from commons.segmentation import AtureTest
from testing import params as p
sep = os.sep
Dirs = {}

Dirs['data'] = 'data' + sep + 'DRIVE' + sep + 'testing'

Dirs['images'] = Dirs['data'] + sep + 'images'
Dirs['mask'] = Dirs['data'] + sep + 'mask'
Dirs['truth'] = Dirs['data'] + sep + '1st_manual'
Dirs['segmented'] = Dirs['data'] + sep + 'segmented_fishing'
Dirs['mst_out'] = Dirs['data'] + sep + 'segmented_mst'

for k, folder in Dirs.items():
    os.makedirs(folder, exist_ok=True)


def get_mask_file(file_name):
    return file_name.split('_')[0] + '_test_mask.gif'


def get_ground_truth_file(file_name):
    return file_name.split('_')[0] + '_manual1.gif'


tester = AtureTest(out_dir=Dirs['mst_out'])
tester.run_all(Dirs=Dirs, save_images=False,
               params_combination=p.get_param_combinations(), fget_mask=get_mask_file, fget_gt=get_ground_truth_file)
