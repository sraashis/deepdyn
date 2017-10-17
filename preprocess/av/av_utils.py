import numpy as np
from matplotlib import pyplot as plt

import preprocess.common.image_utils as img


def get_onh_radius(av_data_set):
    return np.linalg.norm(av_data_set.get_graph('onh')[0] - av_data_set.get_graph('onh_pos'))


def get_av_nodes(av_data_set=None, vessel=""):
    node_position = av_data_set.get_graph('V')
    if vessel.lower() == "art":
        node_marker = av_data_set.get_graph('art')
    if vessel.lower() == "ven":
        node_marker = av_data_set.get_graph('ven')
    av_radius = 2.5 * get_onh_radius(av_data_set=av_data_set)
    xc, yc = av_data_set.get_graph('onh_pos')[0]
    for ix, node in enumerate(node_position):
        x, y = node
        if node_marker[ix] != 1:
            continue
        if ((x - xc) ** 2 + (y - yc) ** 2) < av_radius ** 2:
            yield node


def show_av_graph(av_data_set, image_show=True, onh_show=True, av_show=True):
    onh = av_data_set.get_graph('onh')
    av_art = np.array(list(get_av_nodes(av_data_set, vessel="art")))
    av_ven = np.array(list(get_av_nodes(av_data_set, vessel="ven")))

    av_art = np.ceil(av_art)
    av_ven = np.ceil(av_ven)

    if av_show:
        plt.scatter(av_art[:, 0], av_art[:, 1], color='red', s=1.0)
        plt.scatter(av_ven[:, 0], av_ven[:, 1], color='blue', s=1.0)

    if onh_show:
        plt.plot(onh[:, 0], onh[:, 1], color='yellow')

    if image_show:
        image_array = av_data_set.get_image('I2')
        image_array = img.enhance(image_array, color=0, contrast=3, brightness=0.81, sharpness=4)
        plt.imshow(img.from_array(image_array), aspect='auto')
        plt.show()
