import math as mth

import cv2 as ocv
import matplotlib.pyplot as plt
import numpy as np

import commons.constants as const


def build_filter_bank(k_size, sigma=None, lambd=None, gamma=None, psi=None,
                      k_type=ocv.CV_32F, orientations=None):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / orientations):  # Number of orientations
        params = {'ksize': (k_size, k_size), 'sigma': sigma, 'theta': theta, 'lambd': lambd,
                  'gamma': gamma, 'psi': psi, 'ktype': k_type}
        kern = ocv.getGaborKernel(**params)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def get_chosen_gabor_bank():
    kernels1 = build_filter_bank(k_size=const.GABOR_KERNEL_SIZE1,
                                 gamma=const.GABOR_KERNEL_GAMMA1,
                                 lambd=const.GABOR_KERNEL_LAMBDA1,
                                 sigma=const.GABOR_KERNEL_SIGMA1,
                                 orientations=const.GABOR_KERNEL_NUM_OF_ORIENTATIONS,
                                 psi=const.GABOR_KERNEL_PSI)

    kernels2 = build_filter_bank(k_size=const.GABOR_KERNEL_SIZE2,
                                 gamma=const.GABOR_KERNEL_GAMMA2,
                                 lambd=const.GABOR_KERNEL_LAMBDA2,
                                 sigma=const.GABOR_KERNEL_SIGMA2,
                                 orientations=const.GABOR_KERNEL_NUM_OF_ORIENTATIONS,
                                 psi=const.GABOR_KERNEL_PSI)

    kernels3 = build_filter_bank(k_size=const.GABOR_KERNEL_SIZE3,
                                 gamma=const.GABOR_KERNEL_GAMMA3,
                                 lambd=const.GABOR_KERNEL_LAMBDA3,
                                 sigma=const.GABOR_KERNEL_SIGMA3,
                                 orientations=const.GABOR_KERNEL_NUM_OF_ORIENTATIONS,
                                 psi=const.GABOR_KERNEL_PSI)

    return kernels1 + kernels2 + kernels3


def get_chosen_skeleton_filter():
    kernel = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    return np.array(kernel)


def get_chosen_mask_erode_kernel():
    kern = np.array([
        [0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.2, 1.0, 0.2, 0.0],
        [0.5, 1.0, 1.5, 1.0, 0.5],
        [0.0, 0.2, 1.0, 0.2, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0],
    ], np.uint8)
    return kern


def show_kernels(kernels):
    grid_size = mth.ceil(mth.sqrt(len(kernels)))
    for ix, kernel in enumerate(kernels):
        plt.subplot(grid_size, grid_size, ix + 1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(kernel, cmap='gray', aspect='auto')
    plt.show()


def get_seed_node_list(image_array_2d=None):
    seed = []
    for i in range(image_array_2d.shape[0]):
        for j in range(image_array_2d.shape[1]):
            if image_array_2d[i, j] == 255:
                seed.append((i, j))
    return seed
