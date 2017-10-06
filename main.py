from preprocess import graph_utils as gt
from preprocess import image_utils as img
from preprocess.mat_utils import Mat
import preprocess.av.av_utils as av
from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':
    file = Mat(file_name='wide_image_07.mat')

    image = file.get_image('I2')

    # img.slide_and_construct(image[:, :, 1], m=100, n=100, threshold=.88)

    # img.show_image(image[:, :, 1])

    # auxiliary_graph = file.get_graph('A')
    # node_pos = file.get_graph('V')
    #
    # veins_color = (gt.color_vein(v) for v in file.get_graph('art'))
    # arteries_color = (gt.color_artery(a) for a in file.get_graph('ven'))
    #
    # color = (gt.color_av(a, v) for a, v in zip(file.get_graph('art'), file.get_graph('ven')))
    # onh = file.get_graph('onh')
    #
    # av = np.array(list(av.get_av_nodes(file)))
    #
    # gt.show_graph(auxiliary_matrix=auxiliary_graph, node_pos=node_pos, node_color=''.join(color))
    # plt.scatter(av[:, 0], av[:, 1], color='black')
    # plt.plot(onh[:, 0], onh[:, 1], color='yellow')
    # plt.show()
    av.show_av_graph(file, image_show=True, onh_show=True)
