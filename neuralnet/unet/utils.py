import numpy as np
import utils.img_utils as imgutils


def merge_patches(scores, image_size=(0, 0), training_patch_size=(0, 0)):
    padded_sum = np.zeros([image_size[0], image_size[1]])
    non_zero_count = np.zeros_like(padded_sum)
    for i, chunk_ix in enumerate(imgutils.get_chunk_indexes(image_size, training_patch_size)):
        row_from, row_to, col_from, col_to = chunk_ix
        score = np.exp(scores[i, 1, :, :].squeeze()) * 255
        score_arr = 255 - np.array(score, dtype=np.uint8)
        padded = np.pad(score_arr, [(row_from, image_size[0] - row_to), (col_from, image_size[1] - col_to)],
                        'constant')
        padded_sum = padded + padded_sum
        non_zero_count = non_zero_count + np.array(padded > 0).astype(int)
    return np.array(padded_sum / non_zero_count, dtype=np.uint8)
