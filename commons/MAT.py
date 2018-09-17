"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

from scipy import io


class Mat:
    mat_file = None

    def __init__(self, mat_file=None):
        self.mat_file = io.loadmat(mat_file)

    def get_graph(self, graph_dict_key, root_structure='G'):
        return self.mat_file[root_structure][0][graph_dict_key][()][0]

    def get_image(self, image_dict_key='I2'):
        return self.mat_file[image_dict_key]
