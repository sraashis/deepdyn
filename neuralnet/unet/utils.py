import numpy as np

import utils.img_utils as imgutils


def merge_patches(scores=None, predictions=None, image_size=(0, 0), training_patch_size=(0, 0)):
    padded_sum = np.zeros([image_size[0], image_size[1]])
    non_zero_count = np.zeros_like(padded_sum)
    for i, chunk_ix in enumerate(imgutils.get_chunk_indexes(image_size, training_patch_size)):
        row_from, row_to, col_from, col_to = chunk_ix

        segmented_arr = None
        if scores is not None:
            segmented_arr = np.array(np.exp(scores[i, 1, :, :]).squeeze() * 255, dtype=np.uint8)

        if predictions is not None:
            segmented_arr = np.array(predictions[i, :, :].squeeze(), dtype=np.uint8) * 255

        padded = np.pad(255 - segmented_arr, [(row_from, image_size[0] - row_to), (col_from, image_size[1] - col_to)],
                        'constant')
        padded_sum = padded + padded_sum
        non_zero_count = non_zero_count + np.array(padded > 0).astype(int)
    non_zero_count[non_zero_count == 0] = 1
    return np.array(padded_sum / non_zero_count, dtype=np.uint8)


def get_padding_dims(input_img_size=(388, 388), output_image_size=(572, 572)):
    input_rows, input_cols = input_img_size
    output_rows, output_cols = output_image_size
    print(input_img_size, output_image_size)
    row0 = row1 = (output_rows - input_rows) // 2
    col0 = col1 = (output_cols - input_cols) // 2
    print((row0, row1), (col0, col1))
    return [(row0, row1), (col0, col1)]
