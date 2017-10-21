import os
import numpy as np

import path_config as pth
import preprocess.filter.image_filters as fil
import preprocess.image.image_utils as img
from preprocess.common.mat_utils import Mat

if __name__ == '__main__':
    kernels = fil.build_filter_bank(k_size_start=2, k_size_end=6, k_step=1, gamma=0.5, sigma=0.2, lambd=0.8)
    fil.show_kernels(kernels, save_fig=False)
    files = os.listdir(pth.av_wide_data)

    file = Mat(file_name=str('wide_image_03.mat'))
    image = file.get_image('I2')
    f_image_arr = fil.process(255 - image[:, :, 1], kernels)
    img.from_array(255-f_image_arr).show()

