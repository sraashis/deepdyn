import matplotlib.pyplot as plt
import numpy as np

import math as mth

import cv2 as ocv
import commons.constants as const
import networkx as nx
import PIL.Image as IMG


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


def histogram(image_arr, bins=32):
    plt.hist(image_arr.ravel(), bins)
    plt.show()


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


def show_kernels(kernels):
    grid_size = mth.ceil(mth.sqrt(len(kernels)))
    for ix, kernel in enumerate(kernels):
        plt.subplot(grid_size, grid_size, ix + 1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(kernel, cmap='gray', aspect='auto')
    plt.show()


def assign_cost(graph=nx.Graph(), images=[()], alpha=1, override=False):
    for n1 in graph.nodes():
        for n2 in nx.neighbors(graph, n1):
            if graph[n1][n2] == {} or override:
                cost = 0.0
                for weight, arr in images:
                    m = max(arr[n1[0], n1[1]], arr[n2[0], n2[1]])
                    cost += weight * mth.pow(mth.e, alpha * (m / 255))
                graph[n1][n2]['cost'] = cost


def get_seed_node_list(image_array_2d=None):
    seed = []
    for i in range(image_array_2d.shape[0]):
        for j in range(image_array_2d.shape[1]):
            if image_array_2d[i, j] == 0:
                seed.append((i, j))
    return seed


def as_array(file_name=None):
    original = IMG.open(file_name)
    return np.array(original.getdata(), np.uint8).reshape(original.size[1], original.size[0], 3)
