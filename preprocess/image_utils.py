import os

import numpy as np
from PIL import Image

import app_config as cfg


# @args(2d_array, m, n)
# # 50 * 50 Sliding window by default
def slide_and_construct(nd_array, m=50, n=50, threshold=0.90):
    x = nd_array.shape[0]
    y = nd_array.shape[1]
    x_w = m - 1
    y_w = n - 1
    for i in range(0, x):
        for j in range(0, y):
            print(str(i) + ',' + str(j) + '\n')
            window_x, window_y = i - 1, j - 1
            if window_x < 0:
                window_x = 0
            if window_y < 0:
                window_y = 0
            window_arr = nd_array[window_x:i + x_w, window_y:j + y_w]
            temp_x, temp_y = window_arr.shape[0], window_arr.shape[1]
            avg = np.ndarray.sum(window_arr) / (temp_x * temp_y)
            mx = np.ndarray.max(window_arr)
            mn = np.ndarray.min(window_arr)

            if i - 1 >= 0:
                if nd_array[i - 1, j] >= avg * threshold:
                    nd_array[i - 1, j] = mx
                else:
                    nd_array[i - 1, j] = mn

            if i + 1 <= x - 1:
                if nd_array[i + 1, j] >= avg * threshold:
                    nd_array[i + 1, j] = mx
                else:
                    nd_array[i + 1, j] = mn

            if j - 1 >= 0:
                if nd_array[i, j - 1] >= avg * threshold:
                    nd_array[i, j - 1] = mx
                else:
                    nd_array[i, j - 1] = mn

            if j + 1 <= y - 1:
                if nd_array[i, j + 1] >= avg * threshold:
                    nd_array[i, j + 1] = mx
                else:
                    nd_array[i, j + 1] = mn

            if i + 1 <= x - 1 and j + 1 <= y - 1:
                if nd_array[i + 1, j + 1] >= avg * threshold:
                    nd_array[i + 1, j + 1] = mx
                else:
                    nd_array[i + 1, j + 1] = mn

            if i + 1 <= x - 1 and j - 1 >= 0:
                if nd_array[i + 1, j - 1] >= avg * threshold:
                    nd_array[i + 1, j - 1] = mx
                else:
                    nd_array[i + 1, j - 1] = mn

            if i - 1 >= 0 and j - 1 >= 0:
                if nd_array[i - 1, j - 1] >= avg * threshold:
                    nd_array[i - 1, j - 1] = mx
                else:
                    nd_array[i - 1, j - 1] = mn

            if i - 1 >= 0 and j + 1 <= y - 1:
                if nd_array[i - 1, j + 1] >= avg * threshold:
                    nd_array[i - 1, j + 1] = mx
                else:
                    nd_array[i - 1, j + 1] = mn


def show_image(image_array):
    Image.fromarray(image_array).show()


def save_image(image, x, y):
    new_image = Image.fromarray(image)
    file_name = str(x) + ' by ' + str(y) + '_T_' + '.png'
    os.chdir(cfg.output_path)
    image.save(file_name)
    new_image.show()