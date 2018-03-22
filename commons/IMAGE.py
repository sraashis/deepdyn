import os

import cv2
import cv2 as ocv
import networkx as nx
import numpy as np
import preprocess.utils.img_utils as imgutil
from PIL import Image as IMG

import commons.constants as const
import utils.filter_utils as filutils
import utils.filter_utils as fu
from commons.MAT import Mat
from commons.timer import checktime


class Image:
    def __init__(self):
        self.data_dir = None
        self.file_name = None
        self.image_arr = None
        self.working_arr = None
        self.mask = None
        self.ground_truth = None
        self.res = {}

    def load_file(self, data_dir, file_name):
        self.data_dir = data_dir
        self.file_name = file_name
        self.image_arr = imgutil.get_image_as_array(os.path.join(self.data_dir, self.file_name))

    def load_mask(self, mask_dir=None, fget_mask=None, erode=False):
        try:
            mask_file = fget_mask(self.file_name)
            mask = IMG.open(os.path.join(mask_dir, mask_file))
            mask = np.array(mask.getdata(), np.uint8).reshape(mask.size[1], mask.size[0], 1)[:, :, 0]
            if erode:
                print('Mask loaded: ' + mask_file)
                self.mask = cv2.erode(mask, kernel=fu.get_chosen_mask_erode_kernel(), iterations=5)
        except Exception as e:
            print('Fail to load mask: ' + str(e))
            self.mask = np.ones_like(self.working_arr)

    def apply_mask(self):
        self.working_arr = cv2.bitwise_and(self.working_arr, self.working_arr, mask=self.mask)

    def load_ground_truth(self, gt_dir=None, fget_ground_truth=None):

        try:
            gt_file = fget_ground_truth(self.file_name)
            truth = IMG.open(os.path.join(gt_dir, gt_file))
            truth = np.array(truth.getdata(), np.uint8).reshape(truth.size[1], truth.size[0], 1)[:, :, 0]
            print('Ground truth loaded: ' + gt_file)
            self.ground_truth = truth
        except Exception as e:
            print('Fail to load ground truth: ' + str(e))
            self.ground_truth = np.zeros_like(self.working_arr)


class SegmentedImage(Image):
    def __init__(self):
        super().__init__()

    @checktime
    def apply_bilateral(self, k_size=const.BILATERAL_KERNEL_SIZE, sig_color=const.BILATERAL_SIGMA_COLOR,
                        sig_space=const.BILATERAL_SIGMA_SPACE):
        self.res['bilateral'] = ocv.bilateralFilter(self.working_arr, k_size,
                                                    sigmaColor=sig_color,
                                                    sigmaSpace=sig_space)
        self.res['diff_bilateral'] = imgutil.get_signed_diff_int8(self.working_arr, self.res['bilateral'])
        self.working_arr = self.res['diff_bilateral'].copy()

    @checktime
    def apply_gabor(self, filter_bank=filutils.get_chosen_gabor_bank()):
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
    def generate_lattice_graph(self, eight_connected=const.IMG_LATTICE_EIGHT_CONNECTED):
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


class MatSegmentedImage(SegmentedImage):
    def __init__(self):
        super().__init__()

    def load_file(self, data_dir, file_name):
        self.data_dir = data_dir
        self.file_name = file_name
        file = Mat(mat_file=os.path.join(self.data_dir, self.file_name))
        self.image_arr = file.get_image('I2')


class HighResolutionFundusImage(SegmentedImage):
    def __init__(self):
        super().__init__()

    def load_mask(self, mask_dir=None, fget_mask=None, erode=False):
        try:
            mask_file = fget_mask(self.file_name)
            mask = IMG.open(os.path.join(mask_dir, mask_file))
            mask = np.array(mask.getdata(), np.uint8).reshape(mask.size[1], mask.size[0], 3)[:, :, 0]
            if erode:
                print('Mask loaded: ' + mask_file)
                self.mask = cv2.erode(mask, kernel=fu.get_chosen_mask_erode_kernel(), iterations=5)
        except Exception as e:
            print('Fail to load mask: ' + str(e))
            self.mask = np.ones_like(self.working_arr)
