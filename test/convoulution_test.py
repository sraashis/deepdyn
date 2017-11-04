from multiprocessing import Process

import numpy as np

import preprocess.filter.image_filters as fil
import preprocess.image.image_utils as img


def try_all(image_arr=None, ii=25, jj=5, k_size=4):
    f_name = ''
    for i in range(15, ii):
        for j in range(1, jj):
            for k in np.arange(0.2, 0.6, 0.1):
                try_kernel = fil.build_filter_bank(k_size=k_size, gamma=k, lambd=i, sigma=j)
                f_name = str('size') + str(k_size) + str('_gamma-') + str(k) + ('_lambd-') + str(i) + str(
                    '_sigma-') + str(
                    j) + '.png'
                img_convolved = fil.process(255 - image_arr, try_kernel)
                print('Saving: ' + f_name)
                img.save_image(255 - img_convolved, name=f_name)


def p1(image_array):
    try_all(image_arr=image_array, k_size=4)


def p2(image_array):
    try_all(image_arr=image_array, k_size=32)


def run_test(image_array):
    pr1 = Process(target=p1(image_array))
    pr1.start()
    pr1.join()

    # pr2 = Process(target=p2(image_array))
    # pr2.start()
    # pr2.join()
