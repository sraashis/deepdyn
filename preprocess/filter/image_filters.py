import math as mt
import os
import time

import cv2 as ocv
import matplotlib.pyplot as plt
import numpy as np

import path_config as cfg


def build_filter_bank(k_size_start=3, k_size_end=8, k_step=1, sigma=0.25, lambd=0.80, gamma=0.5, psi=0,
                      k_type=ocv.CV_32F):
    filters = []
    for k_size in np.arange(k_size_start, k_size_end, k_step):
        for theta in np.arange(0, np.pi, np.pi / 9):  # Number of orientations
            sig = k_size * sigma
            lamb = k_size * lambd
            params = {'ksize': (k_size, k_size), 'sigma': sig, 'theta': theta, 'lambd': lamb,
                      'gamma': gamma, 'psi': psi, 'ktype': k_type}
            kern = ocv.getGaborKernel(**params)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


def process(image, filters):
    accumulator = np.zeros_like(image)
    for kern in filters:
        final_image = ocv.filter2D(image, ocv.CV_8UC3, kern)
        np.maximum(accumulator, final_image, accumulator)
    return accumulator


def show_kernels(kernels=None, save_fig=False):
    grid_size = mt.ceil(mt.sqrt(len(kernels)))
    for ix, kernel in enumerate(kernels):
        f_kernel = kernel - np.min(kernel)
        f_kernel = f_kernel / np.max(f_kernel)
        plt.subplot(grid_size, grid_size, ix + 1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(f_kernel * 255, cmap='gray', aspect='auto')
    if save_fig:
        os.chdir(cfg.output_path)
        plt.savefig(str(int(time.time())) + "-gabor_kernels.png")
    plt.show()
