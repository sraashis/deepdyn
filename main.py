import preprocess.filter.image_filters as fil
from preprocess.common.mat_utils import Mat
import preprocess.image.image_utils as img
import matplotlib.pyplot as plt
import test.convoulution_test as cnv
import cv2 as ocv
import numpy as np


if __name__ == '__main__':
    file = Mat(file_name=str('wide_image_03.mat'))
    img_arr = file.get_image('I2')[:, :, 1]
    bi_img = ocv.bilateralFilter(img_arr, 9, 60, 60)
    fin_img = 255 - np.abs((img_arr - bi_img))
    cnv.run_test(fin_img)

