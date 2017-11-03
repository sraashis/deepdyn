from multiprocessing import Process

import preprocess.filter.image_filters as fil
import preprocess.image.image_utils as img
from preprocess.common.mat_utils import Mat


def try_all(image_arr=None, ii=5, jj=10, k_size=4, gamma=0.5):
    f_name = ''
    for i in range(1, ii):
        for j in range(1, jj):
            try_kernel = fil.build_filter_bank(k_size=k_size, gamma=gamma, lambd=i, sigma=j)
            f_name = str('size') + str(k_size) + str('_gamma-') + str(gamma) + ('_lambd-') + str(i) + str(
                '_sigma-') + str(
                j) + '.png'
            img_convolved = fil.process(255 - image_arr, try_kernel)
            print('Saving: ' + f_name)
            img.save_image(255 - img_convolved, name=f_name)


def p1(image_array):
    try_all(image_arr=image_array, ii=20, jj=20, k_size=16)
    try_all(image_arr=image_array, ii=20, jj=20, k_size=20)
    try_all(image_arr=image_array, ii=20, jj=20, k_size=24)


def p2(image_array):
    try_all(image_arr=image_array, ii=20, jj=20, k_size=8)
    try_all(image_arr=image_array, ii=20, jj=20, k_size=10)
    try_all(image_arr=image_array, ii=20, jj=20, k_size=12)


def run_test():
    file = Mat(file_name=str('wide_image_03.mat'))
    image_array = file.get_image('I2')[:, :, 1]

    pr1 = Process(target=p1(image_array))
    pr1.start()
    pr1.join()

    pr2 = Process(target=p2(image_array))
    pr2.start()
    pr2.join()
