import os
import time

from PIL import Image
from PIL import ImageEnhance

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


def enhance(image, color=1, brightness=1, sharpness=1, contrast=1):
    color = ImageEnhance.Color(image).enhance(color)
    contrast = ImageEnhance.Contrast(color).enhance(contrast)
    brightness = ImageEnhance.Brightness(contrast).enhance(brightness)
    sharpness = ImageEnhance.Sharpness(brightness).enhance(sharpness)
    return sharpness
