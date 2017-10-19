import matplotlib.pyplot as plt

import preprocess.filter.image_filters as fil
import preprocess.graph.graph_utils as gt
from preprocess.common.mat_utils import Mat
import preprocess.image.image_utils as img

if __name__ == '__main__':
    file = Mat(file_name='wide_image_03.mat')

    image = file.get_image('I2')

    kernels = fil.build_filter_bank(k_size=24, lambd=5.01, sigma=1.79, psi=0.0, gamma=.89)
    final_image = fil.process(image[:, :, 1], kernels)
    img.show_image(final_image)
    # av.show_av_graph(file)
    # gt.show_vessel_graph(file)
    # plt.show()
