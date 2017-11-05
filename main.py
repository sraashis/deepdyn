import h5py as hdf

import path_config as cfg
import preprocess.av.image_filters as fil

if __name__ == '__main__':
    # file = Mat(file_name=str('wide_image_03.mat'))
    # img_arr = file.get_image('I2')[:, :, 1]
    # bi_img = ocv.bilateralFilter(img_arr, 9, 60, 60)
    # fin_img = 255 - np.abs((img_arr - bi_img))
    # cnv.run_test(fin_img)
    cfg.set_cwd(cfg.out_data_set)
    hdf_kernel = hdf.File("kern.hdf5", "w")
    fil.get_chosen_gabor_bank()


