import h5py as hdf

import path_config as cfg
import preprocess.av.image_filters as fil
from commons.IMAGE import Image

if __name__ == '__main__':
    img = Image('wide_image_03.mat')
    Image.show_image(img.img_array)


