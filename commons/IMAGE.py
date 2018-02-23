import logging as logger

import cv2 as ocv
import numpy as np

import commons.constants as const
import preprocess.utils.img_utils as imgutil
import networkx as nx
from commons.timer import checktime

__all__ = [
    'Image'
]


class Image:
    def __init__(self, image_arr=None, file_name=None):
        self.img_array = image_arr
        self.diff_bilateral = None
        self.img_bilateral = None
        self.img_gabor = None
        self.img_skeleton = None
        self.file_name = file_name
        self.graph = None

    @checktime
    def apply_bilateral(self, k_size=const.BILATERAL_KERNEL_SIZE, sig_color=const.BILATERAL_SIGMA_COLOR,
                        sig_space=const.BILATERAL_SIGMA_SPACE):
        self.img_bilateral = ocv.bilateralFilter(self.img_array, k_size, sigmaColor=sig_color, sigmaSpace=sig_space)
        self.diff_bilateral = imgutil.get_signed_diff_int8(self.img_array, self.img_bilateral)

    @checktime
    def apply_gabor(self, kernel_bank):
        self.img_gabor = np.zeros_like(self.diff_bilateral)
        for kern in kernel_bank:
            final_image = ocv.filter2D(255 - self.diff_bilateral, ocv.CV_8UC3, kern)
            np.maximum(self.img_gabor, final_image, self.img_gabor)
        self.img_gabor = 255 - self.img_gabor

    @checktime
    def create_skeleton(self, threshold=const.SKELETONIZE_THRESHOLD, kernels=None):
        array_2d = self.img_gabor
        self.img_skeleton = np.copy(array_2d)
        self.img_skeleton[self.img_skeleton > threshold] = 255
        self.img_skeleton[self.img_skeleton <= threshold] = 0
        if kernels is not None:
            self.img_skeleton = ocv.filter2D(self.img_skeleton, ocv.CV_8UC3, kernels)

    def _connect_8(graph):
        for i, j in graph:
            n0 = (i, j)
            n1 = (i - 1, j + 1)
            n2 = (i + 1, j - 1)
            n3 = (i - 1, j - 1)
            n4 = (i + 1, j + 1)
            if n1 in graph.nodes():
                graph.add_edge(n0, n1)
            if n2 in graph.nodes():
                graph.add_edge(n0, n2)
            if n3 in graph.nodes():
                graph.add_edge(n0, n3)
            if n4 in graph.nodes():
                graph.add_edge(n0, n4)

    @checktime
    def generate_lattice_graph(self, eight_connected=const.IMG_LATTICE_EIGHT_CONNECTED):
        self.graph = nx.grid_2d_graph(self.img_array.shape[0], self.img_array.shape[1])
        if eight_connected:
            Image._connect_8(self.graph)
