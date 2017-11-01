import math as mt
import os
import time

import cv2 as ocv
import matplotlib.pyplot as plt
import numpy as np

import path_config as cfg
import preprocess.image.image_utils as img


def build_filter_bank(k_size, sigma=2, lambd=5, gamma=0.5, psi=0,
                      k_type=ocv.CV_32F, orientations=16):
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


def try_all(image_arr=None, ii=5, jj=10, k_size=4, gamma=0.5):
    f_name = ''
    for i in range(1, ii):
        for j in range(1, jj):
            try_kernel = build_filter_bank(k_size=k_size, gamma=gamma, lambd=i, sigma=j)
            f_name = str('size') + str(k_size) + str('_gamma-') + str(gamma) + ('_lambd-') + str(i) + str('_sigma-') + str(
                j) + '.png'
            img_convolved = process(255 - image_arr, try_kernel)
            print('Saving: ' + f_name)
            img.save_image(255 - img_convolved, name=f_name)
