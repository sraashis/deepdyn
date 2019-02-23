"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import cv2
import cv2 as ocv
import networkx as nx
import numpy as np

import imgcommons.helper as imgutil
from commons.timer import checktime
from imgcommons import Image


class SegmentedImage(Image):
    def __init__(self):
        super().__init__()

    @checktime
    def apply_bilateral(self, k_size=None, sig_color=None,
                        sig_space=None):
        self.res['bilateral'] = ocv.bilateralFilter(self.working_arr, k_size,
                                                    sigmaColor=sig_color,
                                                    sigmaSpace=sig_space)
        self.res['diff_bilateral'] = imgutil.get_signed_diff_int8(self.working_arr, self.res['bilateral'])
        self.working_arr = self.res['diff_bilateral'].copy()

    @checktime
    def apply_gabor(self, filter_bank=None):
        self.res['gabor'] = np.zeros_like(self.working_arr)
        for kern in filter_bank:
            final_image = ocv.filter2D(255 - self.working_arr, ocv.CV_8UC3, kern)
            np.maximum(self.res['gabor'], final_image, self.res['gabor'])
        self.res['gabor'] = 255 - self.res['gabor']
        self.working_arr = self.res['gabor'].copy()

    def _connect_8(self, graph):
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
    def generate_lattice_graph(self, eight_connected=False):
        self.res['lattice'] = nx.grid_2d_graph(self.working_arr.shape[0], self.working_arr.shape[1])
        if eight_connected:
            self._connect_8(graph=self.res['lattice'])

    @checktime
    def generate_skeleton(self, threshold=100):
        k1 = np.array([[0., 0., 0.],
                       [1., 1., 1.],
                       [0., 0., 0.]], dtype=np.uint8)
        k2 = np.array([[0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.]], dtype=np.uint8)
        binary = 255 - cv2.inRange(self.working_arr, threshold, 255)
        filtered = cv2.erode(binary, k1, iterations=1)
        filtered1 = cv2.erode(binary, k2, iterations=1)
        final = np.maximum(filtered, filtered1)
        self.res['skeleton'] = cv2.bitwise_and(final, final, mask=self.mask)
