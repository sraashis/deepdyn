import numpy as np


def get_onh_radius(av_data_set):
    return np.linalg.norm(av_data_set.get_graph('onh') - av_data_set.get_graph('onh_pos'))


def get_av_nodes(node_position=None, av_data_set=None):

    if node_position is None:
        node_position = av_data_set.get_graph('V')

    av_radius = get_onh_radius(av_data_set)
    xc, yc = av_data_set.get_graph('onh_pos')[0]
    for node in node_position:
        x, y = node
        if ((x - xc) ** 2 + (y - yc) ** 2) < av_radius ** 2:
            yield node
