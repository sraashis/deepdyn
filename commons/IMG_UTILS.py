import os
import time

import matplotlib.pyplot as plt
from PIL import Image

import path_config as cfg
import preprocess.av.av_utils as av
from commons.LOGGER import Logger
from commons.MAT import Mat


class IUtils(Logger):
    def __init__(self, av_file_name, img_key='I2'):
        self.av_file = av_file_name
        self.mat = Mat(file_name=av_file_name)
        self.img_array = self.mat.get_image(img_key)

    @staticmethod
    def from_array(image_array):
        return Image.fromarray(image_array)

    @staticmethod
    def show_image(image_array):
        Image.fromarray(image_array).show()

    @staticmethod
    def save_image(image_array, name="image-" + str(int(time.time()))):
        image = Image.fromarray(image_array)
        file_name = name + '.png'
        os.chdir(cfg.output_path)
        image.save(file_name)

    @staticmethod
    def histogram(image_arr):
        plt.hist(image_arr.ravel(), bins=64)
        plt.show()

    @staticmethod
    def show_av_graph(mat_file=None, image_arr=None, image_show=True, onh_show=True, av_only=True,
                      gray_scale='gray'):
        av.show_av_graph(av_data_set=mat_file, image_array=image_arr, image_show=image_show, onh_show=onh_show,
                         av_only=av_only,
                         gray_scale=gray_scale)
