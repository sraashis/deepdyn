import preprocess.av.av_utils as av
import preprocess.filter.image_filters as fil
import preprocess.image.image_utils as img
from preprocess.common.mat_utils import Mat

if __name__ == '__main__':
    file = Mat(file_name='wide_image_03.mat')

    image = file.get_image('I2')

    # kernels = fil.build_filter_bank(k_size=24, lambd=5.01, sigma=1.79, psi=0.0, gamma=.89)
    # final_image = fil.process(image[:, :, 1], kernels)
    # img.show_image(final_image)
    av.show_av_graph(file, av_only=False,image_show=False)
    # gt.show_vessel_graph(file)
    # plt.show()
