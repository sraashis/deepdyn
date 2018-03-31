import math
import os

import PIL.Image as IMG
import numpy as np


def get_lable(i, j, arr_2d, truth):
    if arr_2d[i, j] == 255 and truth[i, j] == 255:
        return 'white'
    if arr_2d[i, j] == 255 and truth[i, j] == 0:
        return 'green'
    if arr_2d[i, j] == 0 and truth[i, j] == 255:
        return 'red'
    if arr_2d[i, j] == 0 and truth[i, j] == 0:
        return 'black'


def generate_patches(base_path=None, img_obj=None, k_size=51):
    out_dir = os.path.join(base_path, img_obj.file_name.split('.')[0])
    os.makedirs(out_dir, exist_ok=True)
    img = img_obj.working_arr.copy()
    k_half = math.floor(k_size / 2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            patch = np.full((k_size, k_size), 0, dtype=np.uint8)
            patch_exceeds_mask = False

            for k in range(-k_half, k_half + 1, 1):
                for l in range(-k_half, k_half + 1, 1):

                    if patch_exceeds_mask:
                        continue

                    patch_i = i + k
                    patch_j = j + l

                    if patch_i >= 0 and patch_j >= 0 and patch_i < img.shape[0] and patch_j < img.shape[1]:

                        if img_obj.mask[patch_i, patch_j] == 0:
                            patch_exceeds_mask = True

                        patch[k_half + k, k_half + l] = img[patch_i, patch_j]

            if not patch_exceeds_mask:
                IMG.fromarray(patch).save(os.path.join(out_dir,
                                                       str(i) + '_' + str(j) + '_' + get_lable(i, j,
                                                                                               img_obj.res['segmented'],
                                                                                               img_obj.ground_truth) + '.PNG'))
