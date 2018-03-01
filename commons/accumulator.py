import numpy as np


class Accumulator:
    def __init__(self, img_obj):
        self.img_obj = img_obj
        self.x_size, self.y_size = img_obj.working_arr.shape
        self.arr_2d = np.zeros([self.x_size, self.y_size], dtype=np.uint8)
        self.arr_rgb = np.zeros([self.x_size, self.y_size, 3], dtype=np.uint8)
        self.res = {}
