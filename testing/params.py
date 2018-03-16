import itertools as itr
from random import shuffle

import numpy as np


def get_param_combinations():
    SK_THRESHOLD_PARAMS = np.arange(60, 81, 20)
    ALPHA_PARAMS = np.arange(6, 10, 0.5)
    ORIG_CONTRIBUTION_PARAMS = np.arange(0.1, 1.0, 0.2)
    SEGMENTATION_THRESHOLD_PARAMS = np.arange(10, 25, 2)

    PARAMS_ITR = itr.product(SK_THRESHOLD_PARAMS, ALPHA_PARAMS, ORIG_CONTRIBUTION_PARAMS, SEGMENTATION_THRESHOLD_PARAMS)

    PARAMS_COMBINATION = list(PARAMS_ITR)
    shuffle(PARAMS_COMBINATION)

    keys = ('sk_threshold', 'alpha', 'orig_contrib', 'seg_threshold')

    return list(dict(zip(keys, param)) for param in PARAMS_COMBINATION)


print(len(get_param_combinations()))
