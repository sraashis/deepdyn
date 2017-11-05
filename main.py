import h5py as hdf

import path_config as cfg
import preprocess.av.image_filters as fil
from commons.IMAGE import Image

if __name__ == '__main__':
    img = Image('wide_image_03.mat')
    # img.show_image(image_array=img.img_array)
    img.load_kernel_bank()
    img.apply_bilateral(img.img_array[:, :, 1])
    # img.show_image(img.img_bilateral)
    img.apply_gabor(arr=img.img_bilateral, filter_bank=img.kernel_bank)
    img.show_image(img.img_gabor)


