import app_config as cfg
import preprocess.load.av_wide_load as loader
import os

if __name__ == '__main__':
    file = cfg.path(cfg.av_wide_data, 'wide_image_01.mat')
    loader.load_matlab(file)
