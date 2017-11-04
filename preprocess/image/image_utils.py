import os
import time

import matplotlib.pyplot as plt
from PIL import Image

import path_config as cfg


def from_array(image_array):
    return Image.fromarray(image_array)


def show_image(image_array):
    from_array(image_array).show()


def save_image(image_array, name="image-" + str(int(time.time()))):
    image = from_array(image_array)
    file_name = name + '.png'
    os.chdir(cfg.output_path)
    image.save(file_name)


def histogram(image_arr):
    plt.hist(image_arr.ravel(), bins=64)
    plt.show()
