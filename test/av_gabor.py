import os

os.chdir('D:\\idea projects\\pycharm projects\\ature\\')
from commons.IMAGE import Image
import preprocess.av.image_filters as fil

img = Image('wide_image_03.mat')
img.apply_bilateral(img.img_array[:, :, 1])

dif_bilateral = img.get_signed_diff_int8(img.img_array[:, :, 1], img.img_bilateral)

img.from_array(dif_bilateral)

img.apply_gabor(255 - dif_bilateral, filter_bank=fil.get_chosen_gabor_bank())
img.from_array(255 - img.img_gabor)

img.create_skeleton_by_threshold(array_2d=img.img_gabor, threshold=250)

img.from_array(255 - img.img_skeleton)

img.create_skeleton_by_threshold(img.img_gabor, threshold=255)

img.create_lattice_graph(image_arr_2d=255 - img.img_gabor)

images = [(0.25, img.img_array[:, :, 1]), (0.25, img.img_bilateral), (0.5, img.img_gabor)]

img.assign_node_metrics(graph=img.lattice[0], metrics=img.img_skeleton)

img.assign_cost(img.lattice[0], images=images, alpha=15)

g = img.lattice[0]

g[(69, 722)]

import networkx as nx

el = nx.generate_edgelist(g)

en = nx.nodes(g)

n = g.nodes()

n[10]

el = list(el)
nx.minimum_spanning_tree()

(el[100])
