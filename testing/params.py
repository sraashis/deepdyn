import itertools as itr
from random import shuffle

import numpy as np


def get_param_combinations():
    SK_THRESHOLD_PARAMS = [100]
    ALPHA_PARAMS = [5, 5.5, 6]
    ORIG_CONTRIBUTION_PARAMS = [0.6, 0.7, 0.8]
    SEGMENTATION_THRESHOLD_PARAMS = np.arange(10, 15, 1)

    PARAMS_ITR = itr.product(SK_THRESHOLD_PARAMS, ALPHA_PARAMS, ORIG_CONTRIBUTION_PARAMS, SEGMENTATION_THRESHOLD_PARAMS)

    PARAMS_COMBINATION = list(PARAMS_ITR)
    shuffle(PARAMS_COMBINATION)

    keys = ('sk_threshold', 'alpha', 'orig_contrib', 'seg_threshold')

    return list(dict(zip(keys, param)) for param in PARAMS_COMBINATION)

print(len(get_param_combinations()))