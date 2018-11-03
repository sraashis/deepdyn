import os
import threading

import PIL.Image as IMG
import numpy as np


def send_to_back(func, kwargs={}):
    t = threading.Thread(target=func, kwargs=kwargs)
    t.start()


def save_as_img(tensor, to_dir='tensor_img'):
    def f(tsr=tensor, dir=to_dir):
        t = tsr.clone().detach().cpu().numpy() * 255
        t[t < 0] = 0
        t[t > 255] = 255
        os.makedirs(dir, exist_ok=True)
        try:
            for i, b in enumerate(t):
                for j, c in enumerate(b):
                    IMG.fromarray(np.array(c.squeeze(), dtype=np.uint8)).save(
                        dir + os.sep + 't_' + str(i) + '_' + str(j) + '.png')
        except Exception as e:
            print(str(e))

    send_to_back(f)
