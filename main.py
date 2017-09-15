from PIL import Image

import image_utils as img
import mat_utils as mat

if __name__ == '__main__':
    file = mat.Mat('wide_image_03.mat')
    image = file.get_image('I2')
    # Image.fromarray(img.slide_and_construct(image[:, :, 1])).show()
    auxiliary_graph = file.get_graph('A')
    print(auxiliary_graph.shape)

