import numpy as np


# (2d_array, m, n)
# # 3*3 Sliding window by default
def sliding_window(nd_array, m=3, n=3):
    x = nd_array.shape[0]
    y = nd_array.shape[1]
    x_w = m - 1
    y_w = n - 1
    for i in range(0, x):
        for j in range(0, y):
            print(str(i)+','+str(j)+'\n')
            window_x, window_y = i - 1, j - 1
            if window_x < 0:
                window_x = 0
            if window_y < 0:
                window_y = 0
            temp_arr = nd_array[window_x:i + x_w, window_y:j + y_w]
            temp_x, temp_y = temp_arr.shape[0], temp_arr.shape[1]
            avg = np.ndarray.sum(temp_arr) / (temp_x * temp_y)
            mx = np.ndarray.max(temp_arr)

            if i - 1 >= 0:
                if nd_array[i - 1, j] >= avg:
                    nd_array[i - 1, j] = mx
                else:
                    nd_array[i - 1, j] = 0

            if i + 1 <= x - 1:
                if nd_array[i + 1, j] >= avg:
                    nd_array[i + 1, j] = mx
                else:
                    nd_array[i + 1, j] = 0

            if j - 1 >= 0:
                if nd_array[i, j - 1] >= avg:
                    nd_array[i, j - 1] = mx
                else:
                    nd_array[i, j - 1] = 0

            if j + 1 <= y - 1:
                if nd_array[i, j + 1] >= avg:
                    nd_array[i, j + 1] = mx
                else:
                    nd_array[i, j + 1] = 0

            if i + 1 <= x - 1 and j + 1 <= y - 1:
                if nd_array[i + 1, j + 1] >= avg:
                    nd_array[i + 1, j + 1] = mx
                else:
                    nd_array[i + 1, j + 1] = 0

            if i + 1 <= x - 1 and j - 1 >= 0:
                if nd_array[i + 1, j - 1] >= avg:
                    nd_array[i + 1, j - 1] = mx
                else:
                    nd_array[i + 1, j - 1] = 0

            if i - 1 >= 0 and j - 1 >= 0:
                if nd_array[i - 1, j - 1] >= avg:
                    nd_array[i - 1, j - 1] = mx
                else:
                    nd_array[i - 1, j - 1] = 0

            if i - 1 >= 0 and j + 1 <= y - 1:
                if nd_array[i - 1, j + 1] >= avg:
                    nd_array[i - 1, j + 1] = mx
                else:
                    nd_array[i - 1, j + 1] = 0
    return nd_array


if __name__ == '__main__':
    arr = np.array([[3, 2, 3, 8], [5, 5, 9, 7], [10, 11, 8, 6], [4, 5, 9, 10]], np.int32)
    result = sliding_window(arr,4,4)
    print(result)
