import math
import os

import PIL.Image as IMG
import numpy as np
from commons.timer import checktime


def get_lable(i, j, arr_2d, truth):
    if arr_2d[i, j] == 255 and truth[i, j] == 255:
        return 0  # TP White
    if arr_2d[i, j] == 255 and truth[i, j] == 0:
        return 1  # FP Green
    if arr_2d[i, j] == 0 and truth[i, j] == 0:
        return 2  # TN Black
    if arr_2d[i, j] == 0 and truth[i, j] == 255:
        return 3  # FN Red


# Generates patches of images and save in folder with label in name
# Save the images in array and pickle the array. Label is the last element of an array

@checktime
def generate_patches(base_path=None, img_obj=None, k_size=51, save_images=False, pickle=True):
    file_base = img_obj.file_name.split('.')[0]
    out_dir = os.path.join(base_path, file_base)

    if save_images:
        os.makedirs(out_dir, exist_ok=True)

    img = img_obj.working_arr.copy()
    k_half = math.floor(k_size / 2)
    data = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            patch = np.full((k_size, k_size), 0, dtype=np.uint8)
            patch_exceeds_mask = False

            for k in range(-k_half, k_half + 1, 1):

                if patch_exceeds_mask:
                    break

                for l in range(-k_half, k_half + 1, 1):

                    if patch_exceeds_mask:
                        break

                    patch_i = i + k
                    patch_j = j + l

                    if img.shape[0] > patch_i >= 0 and img.shape[1] > patch_j >= 0:
                        if img_obj.mask is not None and img_obj.mask[patch_i, patch_j] == 0:
                            patch_exceeds_mask = True

                        patch[k_half + k, k_half + l] = img[patch_i, patch_j]

            if not patch_exceeds_mask:
                label = get_lable(i, j, img_obj.res['segmented'], img_obj.ground_truth)
                if save_images:
                    IMG.fromarray(patch).save(os.path.join(out_dir, str(i) + '_' + str(j) + '_' + str(label) + '.PNG'))
                if pickle:
                    data.append(np.append(patch.reshape(1, -1), label))

    np.save(os.path.join(base_path, file_base), np.array(data, dtype=np.uint8))
