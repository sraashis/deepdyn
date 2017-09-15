import scipy.io as spio
from PIL import Image

import app_config as cfg
import utils

if __name__ == '__main__':
    file = cfg.path(cfg.av_wide_data, 'wide_image_03.mat')
    img = spio.loadmat(file)
    I2 = img['I2']
    dict_G = img['G']
    A = dict_G['A']
    I2[:, :, 1] = utils.sliding_window(I2[:, :, 1],50,50)
    Image.fromarray(I2[:, :, 1]).save("AV_3_50by50 Window.png")
    # Image.fromarray(A).show()
