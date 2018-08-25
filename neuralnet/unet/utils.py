import numpy as np

import utils.img_utils as imgutils

def get_padding_dims(input_img_size=(388, 388), output_image_size=(572, 572)):
    input_rows, input_cols = input_img_size
    output_rows, output_cols = output_image_size
    print(input_img_size, output_image_size)
    row0 = row1 = (output_rows - input_rows) // 2
    col0 = col1 = (output_cols - input_cols) // 2
    print((row0, row1), (col0, col1))
    return [(row0, row1), (col0, col1)]
