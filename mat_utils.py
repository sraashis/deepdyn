from scipy import io

import app_config as cfg


class Mat:
    mat_file = None
    def __init__(self, file_name):
        self.mat_file = io.loadmat(cfg.path(cfg.av_wide_data, file_name))
        self.image = self.mat_file['I2']

    def get_graph(self, graph_dict_key):
        return self.mat_file['G'][0][graph_dict_key][()][0]

    def get_image(self, image_dict_key='I2'):
        return self.mat_file[image_dict_key]
