import scipy.io as spio
from PIL import Image

import app_config as cfg

if __name__ == '__main__':
    file = cfg.path(cfg.av_wide_data, 'wide_image_01.mat')
    img = spio.loadmat(file)
    I2 = img['I2']
    G_struct = img['G']
    A = G_struct['A'][0]
    print(A.shape)
    # Image.fromarray(I2[:, :, 1], mode='L').show()
    # Image.fromarray(A).show()
