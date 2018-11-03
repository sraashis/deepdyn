import os

import PIL.Image as IMG
import numpy as np


def save_as_img(tensor, to_dir='tensor_img'):
    t = tensor.clone().detach().cpu().numpy() * 255
    t[t < 0] = 0
    t[t > 255] = 255
    os.makedirs(to_dir, exist_ok=True)
    try:
        for i, b in enumerate(t):
            for j, c in enumerate(b):
                IMG.fromarray(np.array(c.squeeze(), dtype=np.uint8)).save(
                    to_dir + os.sep + 't_' + str(i) + '_' + str(j) + '.png')
    except Exception as e:
        print(str(e))
