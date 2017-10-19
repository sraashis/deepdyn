import cv2 as ocv
import numpy as np


def build_filter_bank(k_size=10, sigma=0, lambd=0, gamma=0, psi=0, k_type=ocv.CV_64F):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 7):
        params = {'ksize': (k_size, k_size), 'sigma': sigma, 'theta': theta, 'lambd': lambd,
                  'gamma': gamma, 'psi': psi, 'ktype': k_type}
        kern = ocv.getGaborKernel(**params)
        scaled_kern = kern - np.min(kern)
        scaled_kern = scaled_kern / np.max(scaled_kern)
        filters.append((scaled_kern * 255, params))
    return filters


def process(image, filters):
    accumulator = np.zeros_like(image)
    for kern, params in filters:
        final_image = ocv.filter2D(image, ocv.CV_8UC3, kern)
        np.maximum(accumulator, final_image, accumulator)
    return accumulator
