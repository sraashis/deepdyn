import os

import joblib as jlb
import numpy as np

import path_config as cfg
import preprocess.av.image_filters as fil
import preprocess.av.image_lattice as lat
from commons.IMG_UTILS import IUtils
import networkx as nx


class Image(IUtils):
    kernel_bank = None

    def __init__(self, av_file_name, img_key='I2'):
        Image.log('Loading file ### ' + av_file_name)
        IUtils.__init__(self, av_file_name, img_key='I2')
        self.img_bilateral = None
        self.img_gabor = None
        self.img_skeleton = None
        self.lattice = None

    def apply_bilateral(self, arr=None, k_size=9, sig1=75, sig2=75):
        Image.log('Applying Bilateral filter.')
        if self.img_bilateral is not None:
            Image.warn('Bilateral filter already applied. Overriding...')
        self.img_bilateral = fil.apply_bilateral(arr, k_size=k_size, sig1=sig1, sig2=sig2)

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

    def apply_gabor(self, arr=None, filter_bank=None):
        Image.log('Applying Gabor filter.')
        self.img_gabor = fil.apply_gabor(arr, filter_bank=filter_bank)

    def create_skeleton_by_threshold(self, array_2d=None, threshold=250):
        if self.img_skeleton is not None:
            self.warn('A skeleton already present. Overriding..')
        self.img_skeleton = np.copy(array_2d)
        self.img_skeleton[self.img_skeleton < threshold] = 0

    def create_lattice_graph(self, image_arr_2d=None):
        self.log('Creating 4-connected lattice.')
        if self.lattice is not None:
            self.warn('Lattice already exists. Overriding..')
        self.lattice = lat.create_lattice_graph(image_arr_2d)

    @staticmethod
    def assign_cost(graph=nx.Graph(), images={}, alpha=1, override=False, log=True):
        IUtils.log('Calculating cost of moving to a neighbor.')
        lat.assign_cost(graph, images=images, alpha=alpha, override=override, log=log)

    @staticmethod
    def show_kernel(kernels, save_fig=False):
        fil.show_kernels(kernels, save_fig=save_fig)
