import numpy as np
from matplotlib import pyplot as plt


def get_onh_radius(av_data_set):
    return np.linalg.norm(av_data_set.get_graph('onh')[0] - av_data_set.get_graph('onh_pos'))


def get_av_nodes(av_data_set=None, vessel=""):
    node_position = av_data_set.get_graph('V')

    if vessel == "Art":
        node_position = av_data_set.get_graph('art')

    if vessel == "Ven":
        node_position = av_data_set.get_graph('ven')

    av_radius = 3 * get_onh_radius(av_data_set=av_data_set)
    xc, yc = av_data_set.get_graph('onh_pos')[0]
    for node in node_position:
        x, y = node
        if ((x - xc) ** 2 + (y - yc) ** 2) < av_radius ** 2:
            yield node


def show_av_graph(av_data_set, image_show=False, onh_show=False, av_show=True):
    onh = av_data_set.get_graph('onh')

    av_art = np.array(list(get_av_nodes(av_data_set, vessel="art")))
    av_ven = np.array(list(get_av_nodes(av_data_set, vessel="ven")))

    if av_show:
        plt.scatter(av_art[:, 0], av_art[:, 1], color='blue', s=1.0)
        plt.scatter(av_ven[:, 0], av_ven[:, 1], color='red', s=1.0)

    if onh_show:
        plt.plot(onh[:, 0], onh[:, 1], color='yellow')

    if image_show:
        plt.imshow(av_data_set.get_image('I2')[:, :, :], interpolation='nearest', aspect='auto')

    plt.show()
