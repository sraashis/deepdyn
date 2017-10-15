import os
import matplotlib.pyplot as plt
from PIL import Image
import path_config as cfg


def from_array(image_array):
    return Image.fromarray(image_array)


def show_image(image_array):
    fig = plt.figure()
    plt.imshow(from_array(image_array), aspect='auto')
    fig.tight_layout()
    plt.show()


def save_image(image, x='X', y='Y'):
    new_image = from_array(image)
    file_name = str(x) + ' by ' + str(y) + '_T_' + '.png'
    os.chdir(cfg.output_path)
    new_image.save(file_name)
    new_image.show()


def enhance(image_array, color=1, brightness=1, sharpness=1, contrast=1):
    image_array[:, :, 0] = 0
    image_array[:, :, 2] = 0
    return image_array
