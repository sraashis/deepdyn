import os

os.chdir("D:\\idea projects\\pycharm projects\\ature\\")
from preprocess.common.mat_utils import Mat
import matplotlib.pyplot as plt
import networkx as nx

file = Mat(file_name=str('wide_image_03.mat'))
img_arr = file.get_image('I2')[:, :, 1]

graph_i = nx.grid_2d_graph(img_arr.shape[0], img_arr.shape[1])

node_positions = dict(zip(graph_i.nodes(), graph_i.nodes()))

nx.draw_networkx(graph_i, with_labels=False, pos=node_positions)
plt.show()
