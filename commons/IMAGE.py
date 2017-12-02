import logging as logger
import math as mt

import cv2 as ocv
import matplotlib.pyplot as plt
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
        if self.img_skeleton:
            logger.warning(msg='A skeleton already present. Overriding..')
        self.img_skeleton = np.copy(array_2d)
        self.img_skeleton[self.img_skeleton > threshold] = 255
        self.img_skeleton[self.img_skeleton <= threshold] = 0
        if kernels is not None:
            self.img_skeleton = ocv.filter2D(self.img_skeleton, ocv.CV_8UC3, kernels)

    @staticmethod
    def rescale2d_unsigned(arr):
        m = np.max(arr)
        n = np.min(arr)
        return (arr - n) / (m - n)

    @staticmethod
    def rescale3d_unsigned(arrays):
        return list((Image.rescale2d_unsigned(arr) for arr in arrays))

    @staticmethod
    def get_signed_diff_int8(image_arr1=None, image_arr2=None):
        signed_diff = np.array(image_arr1 - image_arr2, dtype=np.int8)
        fx = np.array(signed_diff - np.min(signed_diff), np.uint8)
        fx = Image.rescale2d_unsigned(fx)
        return np.array(fx * 255, np.uint8)

    @staticmethod
    def show_kernels(kernels):
        grid_size = mt.ceil(mt.sqrt(len(kernels)))
        for ix, kernel in enumerate(kernels):
            plt.subplot(grid_size, grid_size, ix + 1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(kernel, cmap='gray', aspect='auto')
        plt.show()

    @staticmethod
    def histogram(image_arr, bins=32):
        plt.hist(image_arr.ravel(), bins)
        plt.show()

    @staticmethod
    def get_seed_node_list(skeleton_img=None):
        seed = []
        for i in range(skeleton_img.shape[0]):
            for j in range(skeleton_img.shape[1]):
                if skeleton_img[i, j] == 0:
                    seed.append((i, j))
        return seed
