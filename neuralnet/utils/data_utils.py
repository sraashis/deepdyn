import math
import os

import PIL.Image as IMG
import numpy as np

from commons.timer import checktime


def get_dir(i, j, arr_2d, truth):
    if arr_2d[i, j] == 255 and truth[i, j] == 255:
        return 'white'  # TP White
    if arr_2d[i, j] == 255 and truth[i, j] == 0:
        return 'green'  # FP Green
    if arr_2d[i, j] == 0 and truth[i, j] == 0:
        return 'black'  # TN Black
    if arr_2d[i, j] == 0 and truth[i, j] == 255:
        return 'red'  # FN Red


def get_lable(i, j, arr_2d, truth):
    if arr_2d[i, j] == 255 and truth[i, j] == 255:
        return 0  # TP White
    if arr_2d[i, j] == 255 and truth[i, j] == 0:
        return 1  # FP Green
    if arr_2d[i, j] == 0 and truth[i, j] == 0:
        return 2  # TN Black
    if arr_2d[i, j] == 0 and truth[i, j] == 255:
        return 3  # FN Red


@checktime
def generate_patches(base_path=None, img_obj=None, k_size=31, save_images=False, pickle=True):
    """
    :param base_path:
    :param img_obj:
    :param k_size: size of patch
    :param save_images: if True, save patches to specific class folder(Will create folder inside base_path)
    :param pickle: if true, save each patch as flat numpy array pickled file.
            Last element of the array is the label(0,1,2,3)
    :return: None
    """

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
                if save_images:
                    out_path = os.path.join(base_path, get_dir(i, j, img_obj.res['segmented'], img_obj.ground_truth))
                    os.makedirs(out_path, exist_ok=True)
                    patch_name = os.path.join(out_path, img_obj.file_name + ' ' + str(i) + '_' + str(j) + '.PNG')
                    IMG.fromarray(patch).save(patch_name)
                if pickle:
                    data.append(
                        np.append(patch.reshape(-1), get_lable(i, j, img_obj.res['segmented'], img_obj.ground_truth)))

    np.save(base_path, np.array(data, dtype=np.uint8))


def load_dataset(data_path=None, img_shape=None, num_classes=None):
    """
    :param data_path:path to pickled .npy files
    :param img_shape:shape of each patch
    :param num_classes:number of classes
    :return: array of shape(N, *shpae), labels
    """
    data = None
    for data_file in os.listdir(data_path):
        try:
            data_file = os.path.join(data_path, data_file)
            if data is None:
                data = np.load(data_file)
            else:
                data = np.concatenate((data, np.load(data_file)), axis=0)
            print('Data file loaded: ' + data_file)
        except Exception as e:
            print('ERROR loading ' + data_file + ' : ' + str(e))
            continue

    labels = data[:, np.prod(img_shape)]

    if num_classes == 2:
        for i, y in enumerate(labels):
            if y == 0 or y == 3:
                labels[i] = 1
            elif y == 1 or y == 2:
                labels[i] = 0
    data = data[:, 0:np.prod(img_shape)]
    return data.reshape(data.shape[0], *img_shape), labels


def get_class_weights(y):
    """
    :param y: labels
    :return: correct weights of each classes for balanced training
    """
    cls, count = np.unique(y, return_counts=True)
    counter = dict(zip(cls, count))
    majority = max(counter.values())
    return {cls: round(majority / count) for cls, count in counter.items()}
