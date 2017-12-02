import logging as logger

import cv2 as ocv
import numpy as np

__all__ = [
    'Image'
]


class Image:
    def __init__(self, image_arr=None):
        logger.basicConfig(level=logger.INFO)
        self.img_array = image_arr
        self.img_bilateral = None
        self.img_gabor = None
        self.img_skeleton = None

    def apply_bilateral(self, k_size=41, sig_color=20, sig_space=20):
        logger.info(msg='Applying Bilateral filter.')
        self.img_bilateral = ocv.bilateralFilter(self.img_array, k_size, sigmaColor=sig_color, sigmaSpace=sig_space)

    def apply_gabor(self, image_arr, kernel_bank):
        logger.info(msg='Applying Gabor filter.')
        self.img_gabor = np.zeros_like(image_arr)
        for kern in kernel_bank:
            final_image = ocv.filter2D(image_arr, ocv.CV_8UC3, kern)
            np.maximum(self.img_gabor, final_image, self.img_gabor)

    def create_skeleton(self, threshold=0, kernels=None):
        array_2d = 255 - self.img_gabor
        self.img_skeleton = np.copy(array_2d)
        self.img_skeleton[self.img_skeleton > threshold] = 255
        self.img_skeleton[self.img_skeleton <= threshold] = 0
        if kernels is not None:
            self.img_skeleton = ocv.filter2D(self.img_skeleton, ocv.CV_8UC3, kernels)
