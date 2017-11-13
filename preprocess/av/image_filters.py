import math as mt
import os
import time

import cv2 as ocv
import matplotlib.pyplot as plt
import numpy as np

import path_config as cfg


__all__ = [
    "build_filter_bank(k_size, sigma=2, lambd=5, gamma=0.5, psi=0,k_type=ocv.CV_32F, orientations=64)",
    "rescale2d_unsigned(arr)",
    "rescale3d_unsigned(arrays)",
    "process(image, filters)",
    "show_kernels(kernels=None, save_fig=False, file_name=str(int(time.time())))",
    "apply_bilateral(img_arr, k_size=9, sig1=75, sig2=75)",
    "get_signed_diff_int8(image_arr1=None, image_arr2=None)",
    "get_chosen_gabor_bank()",
    "apply_gabor(image_arr, filter_bank=None)"
]


def build_filter_bank(k_size, sigma=2, lambd=5, gamma=0.5, psi=0,
                      k_type=ocv.CV_32F, orientations=64):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / orientations):  # Number of orientations
        params = {'ksize': (k_size, k_size), 'sigma': sigma, 'theta': theta, 'lambd': lambd,
                  'gamma': gamma, 'psi': psi, 'ktype': k_type}
        kern = ocv.getGaborKernel(**params)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def rescale2d_unsigned(arr):
    m = np.max(arr)
    n = np.min(arr)
    return (arr - n) / (m - n)


# Rescale 3d arrays
def rescale3d_unsigned(arrays):
    return list((rescale2d_unsigned(arr) for arr in arrays))


def apply_gabor(image, filters):
    accumulator = np.zeros_like(image)
    for kern in filters:
        final_image = ocv.filter2D(image, ocv.CV_8UC3, kern)
        np.maximum(accumulator, final_image, accumulator)
    return accumulator


def show_kernels(kernels=None, save_fig=False, file_name=str(int(time.time()))):
    grid_size = mt.ceil(mt.sqrt(len(kernels)))
    for ix, kernel in enumerate(kernels):
        plt.subplot(grid_size, grid_size, ix + 1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(kernel, cmap='gray', aspect='auto')
    if save_fig:
        os.chdir(cfg.output_path)
        plt.savefig(file_name + "-gabor_kernels.png")
    else:
        plt.show()


def apply_bilateral(img_arr, k_size=9, sig1=75, sig2=75):
    return ocv.bilateralFilter(img_arr, k_size, sig1, sig2)


def get_signed_diff_int8(image_arr1=None, image_arr2=None):
    signed_diff = np.array(image_arr1 - image_arr2, dtype=np.int8)
    fx = np.array(signed_diff - np.min(signed_diff), np.uint8)
    fx = rescale2d_unsigned(fx)
    return np.array(fx * 255, np.uint8)


def get_chosen_gabor_bank():
    kernels1 = build_filter_bank(k_size=31, gamma=0.7, lambd=5, sigma=2)
    kernels2 = build_filter_bank(k_size=31, gamma=0.7, lambd=7, sigma=3)
    kernels3 = build_filter_bank(k_size=31, gamma=0.7, lambd=10, sigma=4)
    return kernels1 + kernels2 + kernels3
