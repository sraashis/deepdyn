"""
Very useful utilities for working with images
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import copy
import math
import os

import cv2
import numpy as np
from PIL import Image as IMG
from scipy.ndimage.measurements import label

"""
#####################################################################################
A container to hold image related stuffs like paths, mask, ground_truth and many more
#####################################################################################
"""


class Image:

    def __init__(self):
        self.data_dir = None
        self.file_name = None
        self.image_arr = None
        self.working_arr = None
        self.mask = None
        self.ground_truth = None
        self.extra = {}

    def load_file(self, data_dir, file_name):
        try:
            self.data_dir = data_dir
            self.file_name = file_name
            self.image_arr = np.array(IMG.open(os.path.join(self.data_dir, self.file_name)))
        except Exception as e:
            print('### Error Loading file: ' + self.file_name + ': ' + str(e))

    def load_mask(self, mask_dir=None, fget_mask=None):
        try:
            mask_file = fget_mask(self.file_name)
            self.mask = np.array(IMG.open(os.path.join(mask_dir, mask_file)))
        except Exception as e:
            print('### Fail to load mask: ' + str(e))

    def apply_mask(self):
        if self.mask is not None:
            self.working_arr = cv2.bitwise_and(self.working_arr, self.working_arr, mask=self.mask)
        else:
            print('### Mask not applied. ', self.file_name)

    def load_ground_truth(self, gt_dir=None, fget_ground_truth=None):
        try:
            gt_file = fget_ground_truth(self.file_name)
            self.ground_truth = np.array(IMG.open(os.path.join(gt_dir, gt_file)))
        except Exception as e:
            print('Fail to load ground truth: ' + str(e))

    def apply_clahe(self, clip_limit=2.0, tile_shape=(8, 8)):
        enhancer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_shape)
        if len(self.working_arr.shape) == 2:
            self.working_arr = enhancer.apply(self.working_arr)
        elif len(self.working_arr.shape) == 3:
            self.working_arr[:, :, 0] = enhancer.apply(self.working_arr[:, :, 0])
            self.working_arr[:, :, 1] = enhancer.apply(self.working_arr[:, :, 1])
            self.working_arr[:, :, 2] = enhancer.apply(self.working_arr[:, :, 2])
        else:
            print('### More than three channels')

    def __copy__(self):
        copy_obj = Image()
        copy_obj.data_dir = copy.copy(self.data_dir)
        copy_obj.file_name = copy.copy(self.file_name)
        copy_obj.image_arr = copy.copy(self.image_arr)
        copy_obj.working_arr = copy.copy(self.working_arr)
        copy_obj.mask = copy.copy(self.mask)
        copy_obj.ground_truth = copy.copy(self.ground_truth)
        copy_obj.extra = copy.deepcopy(self.extra)
        return copy_obj


"""
##################################################################################################
Very useful image related utilities
##################################################################################################
"""


def get_rgb_scores(arr_2d=None, truth=None):
    """
    Returns a rgb image of pixelwise separation between ground truth and arr_2d
    (predicted image) with different color codes
    Easy when needed to inspect segmentation result against ground truth.
    :param arr_2d:
    :param truth:
    :return:
    """
    arr_rgb = np.zeros([arr_2d.shape[0], arr_2d.shape[1], 3], dtype=np.uint8)
    for i in range(0, arr_2d.shape[0]):
        for j in range(0, arr_2d.shape[1]):
            if arr_2d[i, j] == 255 and truth[i, j] == 255:
                arr_rgb[i, j, :] = 255
            if arr_2d[i, j] == 255 and truth[i, j] == 0:
                arr_rgb[i, j, 0] = 0
                arr_rgb[i, j, 1] = 255
                arr_rgb[i, j, 2] = 0
            if arr_2d[i, j] == 0 and truth[i, j] == 255:
                arr_rgb[i, j, 0] = 255
                arr_rgb[i, j, 1] = 0
                arr_rgb[i, j, 2] = 0
    return arr_rgb


def get_praf1(arr_2d=None, truth=None):
    """
    Returns precision, recall, f1 and accuracy score between two binary arrays upto five precision.
    :param arr_2d:
    :param truth:
    :return:
    """
    x = arr_2d.copy()
    y = truth.copy()
    x[x == 255] = 1
    y[y == 255] = 1
    xy = x + (y * 2)
    tp = xy[xy == 3].shape[0]
    fp = xy[xy == 1].shape[0]
    tn = xy[xy == 0].shape[0]
    fn = xy[xy == 2].shape[0]
    try:
        p = tp / (tp + fp)
    except ZeroDivisionError:
        p = 0

    try:
        r = tp / (tp + fn)
    except ZeroDivisionError:
        r = 0

    try:
        a = (tp + tn) / (tp + fp + fn + tn)
    except ZeroDivisionError:
        a = 0

    try:
        f1 = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1 = 0

    return {
        'Precision': round(p, 5),
        'Recall': round(r, 5),
        'Accuracy': round(a, 5),
        'F1': round(f1, 5)
    }


def rescale2d_unsigned(arr):
    m = np.max(arr)
    n = np.min(arr)
    return (arr - n) / (m - n)


def rescale3d_unsigned(arrays):
    return list(rescale2d_unsigned(arr) for arr in arrays)


def get_signed_diff_int8(image_arr1=None, image_arr2=None):
    signed_diff = np.array(image_arr1 - image_arr2, dtype=np.int8)
    fx = np.array(signed_diff - np.min(signed_diff), np.uint8)
    fx = rescale2d_unsigned(fx)
    return np.array(fx * 255, np.uint8)


def whiten_image2d(img_arr2d=None):
    img_arr2d = img_arr2d.copy()
    img_arr2d = (img_arr2d - img_arr2d.mean()) / img_arr2d.std()
    return np.array(rescale2d_unsigned(img_arr2d) * 255, dtype=np.uint8)


def get_image_as_array(image_file, channels=3):
    img = IMG.open(image_file)
    arr = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], channels)
    if channels == 1:
        arr = arr.squeeze()
    # Sometimes binary image is red as 0 and 1's instead of 255.
    return arr * 255 if np.array_equal(arr, arr.astype(bool)) else arr


def get_chunk_indexes(img_shape=(0, 0), chunk_shape=(0, 0), offset_row_col=None):
    """
    Returns a generator for four corners of each patch within image as specified.
    :param img_shape: Shape of the original image
    :param chunk_shape: Shape of desired patch
    :param offset_row_col: Offset for each patch on both x, y directions
    :return:
    """
    img_rows, img_cols = img_shape
    chunk_row, chunk_col = chunk_shape
    offset_row, offset_col = offset_row_col

    row_end = False
    for i in range(0, img_rows, offset_row):
        if row_end:
            continue
        row_from, row_to = i, i + chunk_row
        if row_to > img_rows:
            row_to = img_rows
            row_from = img_rows - chunk_row
            row_end = True

        col_end = False
        for j in range(0, img_cols, offset_col):
            if col_end:
                continue
            col_from, col_to = j, j + chunk_col
            if col_to > img_cols:
                col_to = img_cols
                col_from = img_cols - chunk_col
                col_end = True
            yield [int(row_from), int(row_to), int(col_from), int(col_to)]


def get_chunk_indices_by_index(img_shape=(0, 0), chunk_shape=(0, 0), indices=None):
    """
    :param img_shape: Original image shape
    :param chunk_shape: Desired patch shape
    :param indices: List of pixel location around which the patch corners will be generated
    :return:
    """
    x, y = chunk_shape
    row_end, col_end = img_shape

    if x % 2 == 0:
        row_from = x / 2 - 1
    else:
        row_from = x // 2

    if y % 2 == 0:
        col_from = y / 2 - 1
    else:
        col_from = y // 2
    row_to, col_to = x // 2 + 1, y // 2 + 1

    for i, j in indices:
        p = i - row_from
        q = i + row_to
        r = j - col_from
        s = j + col_to

        if p < 0 or r < 0:
            continue
        if q > row_end or s > col_end:
            continue
        yield [int(p), int(q), int(r), int(s)]


def merge_patches(patches=None, image_size=(0, 0), patch_size=(0, 0), offset_row_col=None):
    """
    Merge different pieces of image to form a full image. Overlapped regions are averaged.
    :param patches: List of all patches to merge in order (left to right).
    :param image_size: Full image size
    :param patch_size: A patch size(Patches must be uniform in size to be able to merge)
    :param offset_row_col: Offset used to chunk the patches.
    :return:
    """
    padded_sum = np.zeros([image_size[0], image_size[1]])
    non_zero_count = np.zeros_like(padded_sum)
    for i, chunk_ix in enumerate(get_chunk_indexes(image_size, patch_size, offset_row_col)):
        row_from, row_to, col_from, col_to = chunk_ix

        patch = np.array(patches[i, :, :]).squeeze()

        padded = np.pad(patch, [(row_from, image_size[0] - row_to), (col_from, image_size[1] - col_to)],
                        'constant')
        padded_sum = padded + padded_sum
        non_zero_count = non_zero_count + np.array(padded > 0).astype(int)
    non_zero_count[non_zero_count == 0] = 1
    return np.array(padded_sum / non_zero_count, dtype=np.uint8)


def expand_and_mirror_patch(full_img_shape=None, orig_patch_indices=None, expand_by=None):
    """
    Given a patch within an image, this function select a speciified region around it if present, else mirros it.
    It is useful in neuralnetworks like u-net which look for wide range of area than the actual input image.
    :param full_img_shape: Full image shape
    :param orig_patch_indices: Four cornets of the actual patch
    :param expand_by: Expand by (x, y ) in each dimension
    :return:
    """

    i, j = int(expand_by[0] / 2), int(expand_by[1] / 2)
    p, q, r, s = orig_patch_indices
    a, b, c, d = p - i, q + i, r - j, s + j
    pad_a, pad_b, pad_c, pad_d = [0] * 4
    if a < 0:
        pad_a = i - p
        a = 0
    if b > full_img_shape[0]:
        pad_b = b - full_img_shape[0]
        b = full_img_shape[0]
    if c < 0:
        pad_c = j - r
        c = 0
    if d > full_img_shape[1]:
        pad_d = d - full_img_shape[1]
        d = full_img_shape[1]
    return a, b, c, d, [(pad_a, pad_b), (pad_c, pad_d)]


def remove_connected_comp(segmented_img, connected_comp_diam_limit=20):
    """
    Remove connected components of a binary image that are less than smaller than specified diameter.
    :param segmented_img: Binary image.
    :param connected_comp_diam_limit: Diameter limit
    :return:
    """
    img = segmented_img.copy()
    structure = np.ones((3, 3), dtype=np.int)
    labeled, n_components = label(img, structure)
    for i in range(n_components):
        ixy = np.array(list(zip(*np.where(labeled == i))))
        x1, y1 = ixy[0]
        x2, y2 = ixy[-1]
        dst = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if dst < connected_comp_diam_limit:
            for u, v in ixy:
                img[u, v] = 0
    return img


def get_pix_neigh(i, j, eight=False):
    """
    Get four/ eight neighbors of an image.
    :param i: x position of pixel
    :param j: y position of pixel
    :param eight: Eight neighbors? Else four
    :return:
    """

    n1 = (i - 1, j - 1)
    n2 = (i - 1, j)
    n3 = (i - 1, j + 1)
    n4 = (i, j - 1)
    n5 = (i, j + 1)
    n6 = (i + 1, j - 1)
    n7 = (i + 1, j)
    n8 = (i + 1, j + 1)
    if eight:
        return [n1, n2, n3, n4, n5, n6, n7, n8]
    else:
        return [n2, n5, n7, n4]
