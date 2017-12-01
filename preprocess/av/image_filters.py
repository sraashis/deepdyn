import cv2 as ocv
import numpy as np


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


def get_chosen_gabor_bank():
    kernels1 = build_filter_bank(k_size=31, gamma=0.7, lambd=5, sigma=2)
    kernels2 = build_filter_bank(k_size=31, gamma=0.7, lambd=8, sigma=3)
    kernels3 = build_filter_bank(k_size=31, gamma=0.7, lambd=11, sigma=4)
    return kernels1 + kernels2 + kernels3


def get_chosen_skeleton_filter():
    kernel = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    return np.array(kernel)
