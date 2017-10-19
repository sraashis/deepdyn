import preprocess.av.av_utils as av
import preprocess.filter.image_filters as fil
import preprocess.image.image_utils as img
from preprocess.common.mat_utils import Mat
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file = Mat(file_name='wide_image_13.mat')

    image = file.get_image('I2')

    kernels = fil.build_filter_bank(k_size=24, lambd=5.01, sigma=1.79, psi=0.0, gamma=.89)
    # image[:, :, 0] = 0
    # image[:, :, 2] = 0
    final_image = fil.process(image[:, :, 1], kernels)
    plt.imshow(final_image, cmap='gray')
    img.show_image(final_image)
    av.show_av_graph(file, av_only=False,image_show=True)
    plt.show()
