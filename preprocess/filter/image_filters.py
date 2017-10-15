import cv2
import numpy as np


# Gabor Kernel based filter
def build_filters():
    filters = []
    k_size = 31
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {'ksize': (k_size, k_size), 'sigma': 1.0, 'theta': theta, 'lambda': 15.0,
                  'gamma': 0.02, 'psi': 0, 'ktype': cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5 * kern.sum()
        filters.append((kern, params))
    return filters


def process(img, filters):
    accumulator = np.zeros_like(img)
    for kern, params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accumulator, fimg, accumulator)
    return accumulator
