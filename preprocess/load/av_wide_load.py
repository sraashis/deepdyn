import scipy.io as scio
import numpy as np


def load_matlab(file_name):
    file = scio.loadmat(file_name)
    values = file.values()
    print("")

