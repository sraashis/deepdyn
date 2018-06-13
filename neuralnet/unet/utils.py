import numpy as np


def equalize_offset(img_dim, patch_dim):
    extra = img_dim % patch_dim
    if extra == 0:
        return patch_dim
    num_patches = img_dim // patch_dim
    return extra // num_patches


def merge_patches(scores, image_size=(0, 0), training_patch_size=(0, 0)):
    img_rows, img_cols = image_size
    patch_rows, patch_cols = training_patch_size
    rows_offset = 0
    i_to = int(np.ceil(img_rows / patch_rows))
    j_to = int(np.ceil(img_cols / patch_cols))
    padded = np.zeros([img_rows, img_cols, i_to * j_to])
    index = 0

    for i in range(0, img_rows, patch_rows):
        row_from = i - rows_offset
        row_to = i - rows_offset + patch_rows
        if row_to > img_rows:
            row_to = img_rows
            row_from = img_rows - patch_rows
        rows_offset = equalize_offset(img_rows, patch_rows)

        cols_offset = 0
        for j in range(0, img_cols, patch_cols):
            col_from = j - cols_offset
            col_to = j - cols_offset + patch_cols
            if col_to > img_cols:
                col_to = img_cols
                col_from = img_cols - patch_cols
            cols_offset = equalize_offset(img_cols, patch_cols)

            score = np.exp(scores[index, 1, :, :].squeeze()) * 255
            score_arr = 255 - np.array(score, dtype=np.uint8)
            padded[:, :, index] = np.pad(score_arr, [(row_from, img_rows - row_to), (col_from, img_cols - col_to)],
                                         'constant')
            index += 1

    # Find average among non-zero elements in the third dimension
    p = np.sum(padded, axis=2) / np.count_nonzero(padded, axis=2)
    return np.array(p, dtype=np.uint8)
