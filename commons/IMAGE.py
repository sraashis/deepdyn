import os

import joblib as jlb

import path_config as cfg
import preprocess.av.image_filters as fil
from commons.IMG_UTILS import IUtils


class Image(IUtils):
    kernel_bank = None

    def __init__(self, av_file_name, img_key='I2'):
        self.img_bilateral = None
        self.img_gabor = None

    def apply_bilateral(self, arr=None, k_size=9, sig1=75, sig2=75):
        if self.img_bilateral is not None:
            self.warn('Bilateral filter already applied.')
        self.img_bilateral = fil.apply_bilateral(arr, k_size=k_size, sig1=sig1, sig2=sig2)

    def load_kernel_bank(self, kern_file_name='kern.pkl'):
        if self.kernel_bank is not None:
            self.warn('Kernel already loaded.')
        try:
            os.chdir(cfg.kernel_dmp_path)
            self.kernel_bank = jlb.load(kern_file_name)
        except:
            print('[Warn] Cannot load gabor kernel from kern.pkl. Creating new and dumping.')
            self.kernel_bank = fil.get_chosen_gabor_bank()
            jlb.dump(self.kernel_bank, filename=kern_file_name, compress=True)

    def apply_gabor(self, arr=None, bank=None):
        self.img_gabor = fil.apply_gabor(arr, bank=bank)

    @staticmethod
    def show_kernel(self, kernels, save_fig=False):
        fil.show_kernels(kernels, save_fig=save_fig)
