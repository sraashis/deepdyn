"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

from scipy import io
from imgcommons import Image
import os

class Mat:
    mat_file = None

    def __init__(self, mat_file=None):
        self.mat_file = io.loadmat(mat_file)

    def get_graph(self, graph_dict_key, root_structure='G'):
        return self.mat_file[root_structure][0][graph_dict_key][()][0]

    def get_image(self, image_dict_key='I2'):
        return self.mat_file[image_dict_key]


class MatSegmentedImage(Image):
    def __init__(self):
        super().__init__()

    def load_file(self, data_dir, file_name):
        self.data_dir = data_dir
        self.file_name = file_name
        file = Mat(mat_file=os.path.join(self.data_dir, self.file_name))
        self.image_arr = file.get_image('I2')