import image_utils as img
import mat_utils as mat
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    file = mat.Mat('wide_image_03.mat')
    image = file.get_image('I2')
    img.slide_and_construct(image[:, :, 1])
    auxiliary_graph = file.get_graph('A')
    image_graph = np.zeros((auxiliary_graph.shape[0], auxiliary_graph.shape[1], 3), dtype=np.int)
    # rows, cols = auxiliary_graph.nonzero()
    # for row,col in zip(rows, cols):
    #     image_graph[row][col][1] = 255
    # print(image_graph.shape)
    img.show(image[:, :, 1])


