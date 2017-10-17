import preprocess.common.image_utils as img
import preprocess.filter.image_filters as fil
from preprocess.common.mat_utils import Mat

if __name__ == '__main__':
    file = Mat(file_name='wide_image_03.mat')

    image = file.get_image('I2')

    kernels = fil.build_filter_bank(k_size=24, lambd=5.01, sigma=1.79, psi=0.0, gamma=.89)
    final_image = fil.process(image[:, :, 1], kernels)
    img.show_image(final_image)


# --------------- USAGE Examples ----------------
# https://stackoverflow.com/questions/30071474/opencv-getgaborkernel-parameters-for-filter-bank
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
# av.show_av_graph(file)
# av.region_growing(file)
