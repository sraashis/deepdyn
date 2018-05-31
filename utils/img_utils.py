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
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(0, arr_2d.shape[0]):
        for j in range(0, arr_2d.shape[1]):
            if arr_2d[i, j] == 255 and truth[i, j] == 255:
                tp += 1
            if arr_2d[i, j] == 255 and truth[i, j] == 0:
                fp += 1
            if arr_2d[i, j] == 0 and truth[i, j] == 255:
                fn += 1
            if arr_2d[i, j] == 0 and truth[i, j] == 0:
                tn += 1
    p, r, a, f1 = 0, 0, 0, 0
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
