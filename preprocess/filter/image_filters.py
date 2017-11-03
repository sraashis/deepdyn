import math as mt
import os
import time

import cv2 as ocv
import matplotlib.pyplot as plt
import numpy as np

import path_config as cfg


def build_filter_bank(k_size, sigma=2, lambd=5, gamma=0.5, psi=0,
                      k_type=ocv.CV_32F, orientations=32):
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


def apply_chosen_gabor_bank(image_arr):
    kernels0 = build_filter_bank(k_size=16, gamma=0.6, lambd=7, sigma=3)
    kernels1 = build_filter_bank(k_size=24, gamma=0.6, lambd=6, sigma=3)
    kernels2 = build_filter_bank(k_size=32, gamma=0.6, lambd=4, sigma=2)
    kernels3 = build_filter_bank(k_size=32, gamma=0.6, lambd=6, sigma=3)
    kernels4 = build_filter_bank(k_size=64, gamma=0.7, lambd=13, sigma=6)
    kernels = kernels0 + kernels1 + kernels2 + kernels3 + kernels4
    return process(255 - image_arr, kernels)
