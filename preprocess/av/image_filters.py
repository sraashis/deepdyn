import math as mt
import os
import time

import cv2 as ocv
import matplotlib.pyplot as plt
import numpy as np

import path_config as cfg


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


def rescale2d_0_1(arr):
    m = np.max(arr)
    n = np.min(arr)
    return (arr - n) / (m - n)


# Rescale 3d arrays
def rescale3d_0_1(arrays):
    return list((rescale2d_0_1(arr) for arr in arrays))


def process(image, filters):
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


def get_chosen_gabor_bank():
    kernels0 = build_filter_bank(k_size=4, gamma=0.6, lambd=10, sigma=3)
    kernels1 = build_filter_bank(k_size=16, gamma=0.6, lambd=4, sigma=1)
    kernels2 = build_filter_bank(k_size=32, gamma=0.6, lambd=4, sigma=1)
    kernels3 = build_filter_bank(k_size=48, gamma=0.6, lambd=10, sigma=3)
    return kernels0 + kernels1 + kernels2 + kernels3


def apply_gabor(image_arr, filter_bank=None):
    return process(image_arr, filter_bank)
