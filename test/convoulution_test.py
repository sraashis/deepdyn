from multiprocessing import Process

import preprocess.filter.image_filters as fil
from preprocess.common.mat_utils import Mat


def p1(image_array):
    fil.try_all(image_arr=image_array, ii=20, jj=20, k_size=16)
    fil.try_all(image_arr=image_array, ii=20, jj=20, k_size=20)
    fil.try_all(image_arr=image_array, ii=20, jj=20, k_size=24)


def p2(image_array):
    fil.try_all(image_arr=image_array, ii=20, jj=20, k_size=8)
    fil.try_all(image_arr=image_array, ii=20, jj=20, k_size=10)
    fil.try_all(image_arr=image_array, ii=20, jj=20, k_size=12)


def run_test():
    file = Mat(file_name=str('wide_image_03.mat'))
    image_array = file.get_image('I2')[:, :, 1]

    pr1 = Process(target=p1(image_array))
    pr1.start()
    pr1.join()

    pr2 = Process(target=p2(image_array))
    pr2.start()
    pr2.join()
