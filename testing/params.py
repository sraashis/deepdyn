import itertools as itr
from random import shuffle

import numpy as np


def get_param_combinations():
    SK_THRESHOLD_PARAMS = [60]
    ALPHA_PARAMS = np.arange(6, 7.1, 0.5)
    ORIG_CONTRIBUTION_PARAMS = [0.7]
    SEGMENTATION_THRESHOLD_PARAMS = np.arange(14, 25, 2)

    PARAMS_ITR = itr.product(SK_THRESHOLD_PARAMS, ALPHA_PARAMS, ORIG_CONTRIBUTION_PARAMS, SEGMENTATION_THRESHOLD_PARAMS)

    PARAMS_COMBINATION = list(PARAMS_ITR)
    shuffle(PARAMS_COMBINATION)

    keys = ('sk_threshold', 'alpha', 'orig_contrib', 'seg_threshold')

    return list(dict(zip(keys, param)) for param in PARAMS_COMBINATION)
