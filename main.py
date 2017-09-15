import mat_utils as mat
import graph_utils as gt

if __name__ == '__main__':
    file = mat.Mat('wide_image_03.mat')
    image = file.get_image('I2')
    # img.slide_and_construct(image[:, :, 1])
    auxiliary_graph = file.get_graph('A')
    node_pos = file.get_graph('V')
    gt.show_graph(auxiliary_matrix=auxiliary_graph,node_pos=node_pos)
