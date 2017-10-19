import numpy as np
from matplotlib import pyplot as plt

import preprocess.image.image_utils as img


def get_onh_radius(av_data_set):
    return np.linalg.norm(av_data_set.get_graph('onh')[0] - av_data_set.get_graph('onh_pos'))


# Function to get AV region nodes
# Pass 'art' for arteries && 'ven' for veins.
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


def show_av_graph(av_data_set, image_show=True, onh_show=True, av_only=True):

    onh = av_data_set.get_graph('onh')
    av_art = np.array(list(get_av_nodes_positions(av_data_set, vessel="art", av_only=av_only)))
    av_ven = np.array(list(get_av_nodes_positions(av_data_set, vessel="ven", av_only=av_only)))

    av_art = np.ceil(av_art)
    av_ven = np.ceil(av_ven)

    plt.scatter(av_art[:, 0], av_art[:, 1], color='red', s=1.0)
    plt.scatter(av_ven[:, 0], av_ven[:, 1], color='blue', s=1.0)

    if onh_show:
        plt.plot(onh[:, 0], onh[:, 1], color='green')

    if image_show:
        image_array = av_data_set.get_image('I2')
        plt.imshow(img.from_array(image_array), aspect='auto')
    plt.show()

