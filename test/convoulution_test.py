from multiprocessing import Process

import preprocess.filter.image_filters as fil
from preprocess.common.mat_utils import Mat


def p1(image_array):
    fil.try_all(image_arr=image_array[:, :, 1], ii=30, jj=30, gamma=0.5, k_size=3)
    fil.try_all(image_arr=image_array[:, :, 1], ii=30, jj=30, gamma=0.5, k_size=4)
    fil.try_all(image_arr=image_array[:, :, 1], ii=30, jj=30, gamma=0.5, k_size=5)


def p2(image_array):
    fil.try_all(image_arr=image_array[:, :, 1], ii=30, jj=30, gamma=0.5, k_size=6)
    fil.try_all(image_arr=image_array[:, :, 1], ii=30, jj=30, gamma=0.5, k_size=7)
    fil.try_all(image_arr=image_array[:, :, 1], ii=30, jj=30, gamma=0.5, k_size=8)
    fil.try_all(image_arr=image_array[:, :, 1], ii=30, jj=30, gamma=0.6, k_size=12)


def p3(image_array):
    fil.try_all(image_arr=image_array[:, :, 1], ii=30, jj=30, gamma=0.6, k_size=16)
    fil.try_all(image_arr=image_array[:, :, 1], ii=30, jj=30, gamma=0.6, k_size=24)
    fil.try_all(image_arr=image_array[:, :, 1], ii=30, jj=30, gamma=0.6, k_size=32)
    fil.try_all(image_arr=image_array[:, :, 1], ii=30, jj=30, gamma=0.7, k_size=64)


def run_test():
    file = Mat(file_name=str('wide_image_03.mat'))
    image_array = file.get_image('I2')

    pr1 = Process(target=p1(image_array))
    pr1.start()
    pr1.join()

    pr2 = Process(target=p2(image_array))
    pr2.start()
    pr2.join()

    pr3 = Process(p3(image_array))
    pr3.start()
    pr3.join()
