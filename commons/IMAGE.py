import os

import joblib as jlb
import numpy as np

import path_config as cfg
import preprocess.av.image_filters as fil
from commons.timer import check_time

__all__ = [
    'Image'
]


class Image:
    __all__ = [
        '__init__(self, av_file_name, img_key=\'I2\'',
        'apply_bilateral(self, arr=None, k_size=41, sig1=20, sig2=20)',
        'get_signed_diff_int8(image_arr1=None, image_arr2=None)',
        'load_kernel_bank(self, kern_file_name=\'kernel_bank.pkl\')',
        'apply_gabor(self, arr=None, filter_bank=None)',
        'create_skeleton_by_threshold(self, array_2d=None, threshold=250)',
        'assign_cost(graph=nx.Graph(), images=[()], alpha=10, override=False, log=True)',
        'assign_node_metrics(graph=nx.Graph(), metrics=np.ndarray((0, 0)))',
        'show_kernel(kernels, save_fig=False)'
    ]

    kernel_bank = None

    def __init__(self, image_arr):
        self.img_array = image_arr
        self.img_bilateral = None
        self.img_gabor = None
        self.img_skeleton = None

    @check_time
    def apply_bilateral(self, arr=None, k_size=41, sig1=20, sig2=20):
        Image.log('Applying Bilateral filter.')
        if self.img_bilateral is not None:
            Image.warn('Bilateral filter already applied. Overriding...')
        self.img_bilateral = fil.apply_bilateral(arr, k_size=k_size, sig1=sig1, sig2=sig2)

    @staticmethod
    def get_signed_diff_int8(image_arr1=None, image_arr2=None):
        return fil.get_signed_diff_int8(image_arr1=image_arr1, image_arr2=image_arr2)

    @check_time
    def load_kernel_bank(self, kern_file_name='kernel_bank.pkl'):
        Image.log('Loading filter kernel bank.')
        if Image.kernel_bank is not None:
            self.warn('Kernel already loaded. Overriding...')
        try:
            os.chdir(cfg.kernel_dmp_path)
            Image.kernel_bank = jlb.load(kern_file_name)
        except:
            Image.warn('Cannot load gabor kernel from kern.pkl. Creating new and dumping.')
            Image.kernel_bank = fil.get_chosen_gabor_bank()
            jlb.dump(Image.kernel_bank, filename=kern_file_name, compress=True)

    @check_time
    def apply_gabor(self, arr, filter_bank):
        Image.log('Applying Gabor filter.')
        self.img_gabor = fil.apply_gabor(arr, filter_bank)

    def create_skeleton_by_threshold(self, array_2d=None, threshold=5):
        array_2d = 255 - array_2d
        if self.img_skeleton is not None:
            self.warn('A skeleton already present. Overriding..')
        self.img_skeleton = np.copy(array_2d)
        self.img_skeleton[self.img_skeleton > threshold] = 255
        self.img_skeleton[self.img_skeleton <= threshold] = 0

    @staticmethod
    def show_kernel(kernels, save_fig=False):
        fil.show_kernels(kernels, save_fig=save_fig)
