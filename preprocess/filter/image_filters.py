import cv2
import numpy as np

import preprocess.image.image_utils as img


def build_filter_bank(k_size=10, sigma=0, lambd=0, gamma=0, psi=0, k_type=cv2.CV_64F):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / k_size + 1):
        params = {'ksize': (k_size, k_size), 'sigma': sigma, 'theta': theta, 'lambd': lambd,
                  'gamma': gamma, 'psi': psi, 'ktype': k_type}
        kern = cv2.getGaborKernel(**params)
        kern /= 0.93 * kern.sum()
        filters.append((kern, params))
    return filters


def process(image, filters):
    accumulator = np.zeros_like(image)
    for kern, params in filters:
        final_image = cv2.filter2D(image, cv2.CV_8UC3, kern)
        np.maximum(accumulator, final_image, accumulator)
    return accumulator
