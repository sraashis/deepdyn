import PIL.Image as IMG
import matplotlib.pyplot as plt
import numpy as np


def get_rgb_scores(arr_2d=None, truth=None):
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
        'Precision': p,
        'Recall': r,
        'Accuracy': a,
        'F1': f1
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


def histogram(image_arr, bins=32):
    plt.hist(image_arr.ravel(), bins)
    plt.show()


def get_image_as_array(image_file, channels=3):
    img = IMG.open(image_file)
    arr = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], channels)
    if channels == 1:
        arr = arr.squeeze()
    # Sometimes binary image is red as 0 and 1's instead of 255.
    return arr * 255 if np.array_equal(arr, arr.astype(bool)) else arr


def get_chunk_indexes(img_shape=(0, 0), chunk_shape=(0, 0), offset_row_col=None):
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


def merge_patches(patches=None, image_size=(0, 0), patch_size=(0, 0), offset_row_col=None):
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


def get_expanded_patch_indices(orig_shape=None, orig_patch_indices=None, expand_by=None):
    x, y = orig_shape
    i, j = int(expand_by[0] / 2), int(expand_by[1] / 2)
    p, q, r, s = orig_patch_indices
    a = p - i
    b = q + i
    c = r - j
    d = s + j
    pad_a, pad_b, pad_c, pad_d = [0] * 4
    if a < 0:
        pad_a = abs(a)
        a = p
    if b > x:
        pad_b = abs(b)
        b = q
    if c < 0:
        pad_c = abs(c)
        c = r
    if d > y:
        pad_d = abs(d)
        d = s

    return a, b, c, d, [(pad_a, pad_b), (pad_c, pad_d)]
