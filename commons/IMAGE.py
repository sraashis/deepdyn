import os

import cv2
import cv2 as ocv
import networkx as nx
import numpy as np
from PIL import Image as IMG

import commons.constants as const
import preprocess.utils.filter_utils as filutils
import preprocess.utils.filter_utils as fu
import preprocess.utils.img_utils as imgutil
from commons.MAT import Mat
from commons.timer import checktime


class Image:
    def __init__(self):
        self.data_dir = None
        self.file_name = None
        self.image_arr = None
        self.working_arr = None
        self.diff_bilateral = None
        self.img_bilateral = None
        self.img_gabor = None
        self.img_skeleton = None
        self.graph = None
        self.mask = None
        self.ground_truth = None

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

    @checktime
    def apply_bilateral(self):
        self.img_bilateral = ocv.bilateralFilter(self.working_arr, const.BILATERAL_KERNEL_SIZE,
                                                 sigmaColor=const.BILATERAL_SIGMA_COLOR,
                                                 sigmaSpace=const.BILATERAL_SIGMA_SPACE)
        self.diff_bilateral = imgutil.get_signed_diff_int8(self.working_arr, self.img_bilateral)

    @checktime
    def apply_gabor(self):
        self.img_gabor = np.zeros_like(self.diff_bilateral)
        for kern in filutils.get_chosen_gabor_bank():
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
        self.graph = nx.grid_2d_graph(self.working_arr.shape[0], self.working_arr.shape[1])
        if eight_connected:
            Image._connect_8(graph=self.graph)


class MatImage(Image):
    def __init__(self):
        super().__init__()

    def _load(self, file_name=None):
        file = Mat(mat_file=file_name)
        orig = file.get_image('I2')
        print('### File loaded: ' + file_name)
        return orig


class GlaucomaImage(Image):
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

    @checktime
    def apply_bilateral(self):
        self.img_bilateral = ocv.bilateralFilter(self.working_arr, 81,
                                                 sigmaColor=60,
                                                 sigmaSpace=60)
        self.diff_bilateral = imgutil.get_signed_diff_int8(self.working_arr, self.img_bilateral)

    @checktime
    def apply_gabor(self):
        self.img_gabor = np.zeros_like(self.diff_bilateral)
        for kern in filutils.get_chosen_gabor_bank_hd():
            final_image = ocv.filter2D(255 - self.diff_bilateral, ocv.CV_8UC3, kern)
            np.maximum(self.img_gabor, final_image, self.img_gabor)
        self.img_gabor = 255 - self.img_gabor
