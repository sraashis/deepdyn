

import os
import sys

sys.path.append('/home/ak/PycharmProjects/ature')
os.chdir('/home/ak/PycharmProjects/ature')

from testing.segmentation import AtureTest

sep = os.sep
Dirs = {}

Dirs['data'] = 'data' + sep + 'DRIVE' + sep + 'testing'

Dirs['images'] = Dirs['data'] + sep + 'images'
Dirs['mask'] = Dirs['data'] + sep + 'mask'
Dirs['truth'] = Dirs['data'] + sep + '1st_manual'
Dirs['segmented'] = 'data/DRIVE/images'
Dirs['mst_out'] = 'data/DRIVE/mst_best'

for k, folder in Dirs.items():
    os.makedirs(folder, exist_ok=True)


def get_mask_file(file_name):
    return file_name.split('_')[0] + '_test_mask.gif'


def get_ground_truth_file(file_name):
    return file_name.split('_')[0] + '_manual1.gif'


params = {'sk_threshold': 100,
          'alpha': 5.0,
          'orig_contrib': 0.6,
          'seg_threshold': 12}

tester = AtureTest(out_dir=Dirs['mst_out'])
tester.run_all(Dirs=Dirs, save_images=True,
               params_combination=[params], fget_mask=get_mask_file, fget_gt=get_ground_truth_file)
