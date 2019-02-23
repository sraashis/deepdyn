"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

"""
A utility function for av related tasks.
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import networkx as nx


def get_onh_radius(av_data_set):
    return np.linalg.norm(av_data_set.get_graph('onh')[0] - av_data_set.get_graph('onh_pos'))


def get_av_nodes_positions(av_data_set=None, vessel="", av_only=True):
    node_marker = av_data_set.get_graph(vessel)
    av_radius = 2.5 * get_onh_radius(av_data_set)
    xc, yc = av_data_set.get_graph('onh_pos')[0]
    for ix, node in enumerate(av_data_set.get_graph('V')):
        x, y = node
        if node_marker[ix] != 1:
            continue
        if ((x - xc) ** 2 + (y - yc) ** 2) < av_radius ** 2 or not av_only:
            yield node


def show_av_graph(av_data_set=None, image_array=None, image_show=True, onh_show=True, av_only=True, gray_scale=None):
    onh = av_data_set.get_graph('onh')
    av_art = np.array(list(get_av_nodes_positions(av_data_set, vessel="art", av_only=av_only)))
    av_ven = np.array(list(get_av_nodes_positions(av_data_set, vessel="ven", av_only=av_only)))

    av_art = np.ceil(av_art)
    av_ven = np.ceil(av_ven)

    plt.scatter(av_art[:, 0], av_art[:, 1], color='red', s=4.0)
    plt.scatter(av_ven[:, 0], av_ven[:, 1], color='blue', s=4.0)

    if onh_show:
        plt.plot(onh[:, 0], onh[:, 1], color='green')

    if image_show:
        plt.imshow(Image.fromarray(image_array), aspect='auto', cmap=gray_scale)
    plt.show()


def show_graph(adj_matrix, node_pos=None, node_color='red', edge_color='black'):
    graph = nx.from_scipy_sparse_matrix(adj_matrix)
    nx.draw_networkx(graph, pos=node_pos, edge_color=edge_color, node_color=node_color, with_labels=False, node_size=4,
                     width=0.5)
    plt.show()


def color_artery(x): return x == 1 and 'r' or 'b'


def color_vein(x): return x == 1 and 'b' or 'r'


def color_av(a, v):
    if a == 1 and v == 1:
        return 'g'
    if a == 1 and v == 0:
        return 'b'
    if a == 0 and v == 1:
        return 'r'
    return 'g'