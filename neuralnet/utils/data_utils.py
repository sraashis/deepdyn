import os

import numpy as np

sep = os.sep


def get_class_weights(y):
    """
    :param y: labels
    :return: correct weights of each classes for balanced training
    """
    cls, count = np.unique(y, return_counts=True)
    counter = dict(zip(cls, count))
    majority = max(counter.values())
    return {cls: round(majority / count) for cls, count in counter.items()}


def get_4_flips(img_obj=None):
    flipped = [img_obj]
    copy0 = img_obj.__copy__()
    copy0.working_arr = np.flip(copy0.working_arr, 0)

    if copy0.ground_truth is not None:
        copy0.ground_truth = np.flip(copy0.ground_truth, 0)
    if copy0.mask is not None:
        copy0.mask = np.flip(copy0.mask, 0)
    flipped.append(copy0)

    copy1 = copy0.__copy__()
    copy1.working_arr = np.flip(copy1.working_arr, 1)

    if copy1.ground_truth is not None:
        copy1.ground_truth = np.flip(copy1.ground_truth, 1)

    if copy1.mask is not None:
        copy1.mask = np.flip(copy1.mask, 1)
    flipped.append(copy1)

    copy2 = copy1.__copy__()
    copy2.working_arr = np.flip(copy2.working_arr, 0)

    if copy2.ground_truth is not None:
        copy2.ground_truth = np.flip(copy2.ground_truth, 0)

    if copy2.mask is not None:
        copy2.mask = np.flip(copy2.mask, 0)
    flipped.append(copy2)

    return flipped
