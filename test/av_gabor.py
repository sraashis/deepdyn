import networkx as nx
import path_config as cfg

from commons.IMAGE import Image
img = Image('wide_image_03.mat')
# img.show_image(image_array=img.img_array)
img.load_kernel_bank()
img.apply_bilateral(img.img_array[:, :, 1])
# img.show_image(img.img_bilateral)
img_temp = img.img_array[:, :, 1] - img.img_bilateral
img.apply_gabor(arr=img_temp, filter_bank=img.kernel_bank)
img.histogram(img.img_gabor)
# img.img_gabor[img.img_gabor<250] = 0
# img.from_array(255-img.img_gabor)
# img.show_av_graph(mat_file=img.mat, image_arr=img.img_gabor, av_only=False)

img.create_lattice_graph(image_arr_2d=img.img_array[:,:,1])
img.create_skeleton_by_threshold(array_2d=img.img_gabor, threshold=250)
images = {0.5:255-img.img_skeleton,0.25:img.img_array[:,:,1], 0.25:255-img.img_gabor}
img.assign_cost(images=images,alpha=0.1,graph=img.lattice[0])
nx.write_graphml(img.lattice[0], path=cfg.output_path+'\\graph\\one.graphml')

