from scipy import io
import path_config as cfg


class Mat:
    mat_file = None

    def __init__(self, root_path=cfg.av_wide_data, file_name=None):
        self.mat_file = io.loadmat(cfg.join(root_path, file_name))

    def get_graph(self, graph_dict_key, root_structure='G'):
        return self.mat_file[root_structure][0][graph_dict_key][()][0]

    def get_image(self, image_dict_key='I2'):
        return self.mat_file[image_dict_key]
