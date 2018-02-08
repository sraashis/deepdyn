##############################################################################
# IMPORTS, FLAGS, AND FOLDERS
##############################################################################
import os
import numpy as np
from PIL import Image as IMG
import time
import matplotlib.pyplot as plt
import networkx as nx

# Folders and files
Dirs = {}
Dirs['base'] = '/home/rolando/Research/ature2/'
# /home/rolando/Research/ature2/DRIVE/training/images
Dirs['data'] = Dirs['base'] + 'data/av_wide_data_set/'

Files = {}
Files['matFile'] = 'wide_image_02.mat'

# Local imports
os.chdir(Dirs['base'])
from commons.IMAGE import Image
from commons.ImgLATTICE_rje import Lattice
import preprocess.utils.image_filters as fil
import preprocess.utils.image_utils as imgutil
import preprocess.utils.lattice_utils as lat
from commons.MAT import Mat
import preprocess.algorithms.rje_mst as mst

# Execution flags
Flags = {}
Flags['loadImage'] =        True
Flags['filterImage'] =      True
Flags['createLattice'] =    False #True
Flags['runMST'] =           False #True
Flags['getConnComps'] =     True
Flags['checkRunningTime'] = False #True

##############################################################################
# LOAD SAMPLE IMAGE
##############################################################################
if Flags['loadImage']:
    t = time.time()    
    file = Mat(file_name=Dirs['data']+Files['matFile'])
    # original = file.get_image('I2')[300:400,500:600,1]
    # original = file.get_image('I2')[300:700,100:700,1]
    original = file.get_image('I2')[:,:,1]
    img = Image(image_arr=original)
    elapsed = time.time() - t
    print(''.join(('Finished loading image in ',str(elapsed))))


##############################################################################
# FILTER IMAGE
##############################################################################
if Flags['filterImage']:
    t = time.time()    
    img.apply_bilateral()
    img.apply_gabor(kernel_bank= fil.get_chosen_gabor_bank())
    img.create_skeleton(kernels=fil.get_chosen_skeleton_filter(),threshold=5)
    elapsed = time.time() - t
    print(''.join(('Finished filtering image in ',str(elapsed))))


##############################################################################
# CREATE LATTICE
##############################################################################
if Flags['createLattice']:
    t = time.time()    
    lattice = Lattice(image_arr_2d=img.img_gabor)
    lattice.generate_lattice_graph(eight_connected=False)
    images =  [(0.7, img.img_gabor),(0.3, img.img_array)]
    lattice.assign_cost(images=images,alpha=5,threshold=np.inf,log=False,override=True)
    elapsed = time.time() - t
    print(''.join(('Finished creating lattice in ',str(elapsed))))


##############################################################################
# RUN MST
##############################################################################
if Flags['runMST']:
    t = time.time()
    seed_node_list = lat.get_seed_node_list(img.img_skeleton)
    minSpanTree = mst.minimum_spanning_tree(lattice.lattice)
    # mst.run_mst(lattice_object=lattice,seed=seed_node_list,threshold=7.5)
    # IMG.fromarray(lattice.accumulator)
    elapsed = time.time() - t
    print(''.join(('Finished running MST in ',str(elapsed))))


##############################################################################
# GET CONNECTED COMPONENTS
##############################################################################
if Flags['getConnComps']:
        # sx = 500
        # sy = 500
        t = time.time()    
        # lattice = Lattice(image_arr_2d=img.img_gabor[0:sx,0:sy])
        lattice = Lattice(image_arr_2d=img.img_gabor)
        lattice.generate_lattice_graph(eight_connected=False)
        # images =  [(0.7, img.img_gabor[0:sx,0:sy]),(0.3, img.img_array[0:sx,0:sy])]
        images =  [(0.7, img.img_gabor),(0.3, img.img_array)]
        # lattice.assign_cost(images=images,alpha=5,threshold=np.inf,log=False,override=True)
        lattice.assign_cost(images=images,alpha=5,threshold=7.5,log=False,override=True)
        # lattice.assign_cost(images=images,alpha=5,threshold=10,log=False,override=True)

        seed_node_list = lat.get_seed_node_list(img.img_skeleton)
        T, remNodes = mst.valid_conn_comps(lattice.lattice,seed_node_list)

        for node in T.nodes():
            lattice.accumulator[node[0]][node[1]] = 255

        elapsed = time.time() - t
        print(''.join(('Finished running CC in ',str(elapsed))))

##############################################################################
# CHECK RUNNING TIME
##############################################################################
if Flags['checkRunningTime']:
    nSizes = 5
    sizesX = np.round(np.linspace(100,500, nSizes)).astype(np.uint64)
    sizesY = np.round(np.linspace(100,500, nSizes)).astype(np.uint64)
    # nSizes = 10
    # sizesX = np.round(np.linspace(10,100, nSizes)).astype(np.uint64)
    # sizesY = np.round(np.linspace(10,100, nSizes)).astype(np.uint64)
    timesLattice = np.zeros(nSizes)
    timesMST = np.zeros(nSizes)
    for i in range(nSizes):
        sx = sizesX[i]
        sy = sizesY[i]
        t = time.time()    
        lattice = Lattice(image_arr_2d=img.img_gabor[0:sx,0:sy])
        lattice.generate_lattice_graph(eight_connected=False)
        images =  [(0.7, img.img_gabor[0:sx,0:sy]),(0.3, img.img_array[0:sx,0:sy])]
        lattice.assign_cost(images=images,alpha=5,threshold=7.5,log=False,override=True)
        timesLattice[i] = time.time() - t

    #     t = time.time()
    #     seed_node_list = lat.get_seed_node_list(img.img_skeleton)
    #     for comp in nx.connected_components(lattice.lattice):

    # #     minSpanTree = mst.minimum_spanning_tree(lattice.lattice)
    # #     # nx.minimum_spanning_tree(lattice.lattice,algorithm='kruskal')
    # #     # nx.minimum_spanning_tree(lattice.lattice,algorithm='kruskal')
    # #     # nx.prim_mst(lattice)
    # #     # seed_node_list = lat.get_seed_node_list(img.img_skeleton)
    # #     # mst.run_mst(lattice_object=lattice,seed=seed_node_list,threshold=7.5)
    # #     # IMG.fromarray(lattice.accumulator)
    #     timesMST[i] = time.time() - t

    # #O(n)
    # plt.plot(sizesX*sizesY,timesLattice,'.-')
    # plt.show()

    # # O(n)
    # plt.plot(sizesX*sizesY,timesMST,'.-')
    # plt.show()

    # # O(n*log(n))
    # plt.plot(sizesX*sizesY*np.log(sizesX*sizesY),timesMST,'.-')
    # plt.show()

    # # O(n^2)
    # plt.plot((sizesX*sizesY)**2,timesMST,'.-')
    # plt.show()

    # # O(n^3)
    # plt.plot((sizesX*sizesY)**3,timesMST,'.-')
    # plt.show()

    # t = time.time()
    # seed_node_list = lat.get_seed_node_list(img.img_skeleton)
    # mst.run_mst(lattice_object=lattice,seed=seed_node_list,threshold=7.5)
    # IMG.fromarray(lattice.accumulator)
    # elapsed = time.time() - t
    # print(''.join(('Finished running MST in ',str(elapsed))))




    # img.from_array(255-lattice.accumulator)
    # x= 255-img.img_gabor
    # img.from_array(img.img_array-lattice.accumulator)
    # len(seed_node_list)
    # lattice.total_weight


# img = Image(image_arr=original)


# # In[5]:


# IMG.fromarray(img.img_array)


# # In[6]:


# img.apply_bilateral()


# # In[7]:


# IMG.fromarray(img.diff_bilateral)


# # In[8]:


# IMG.fromarray(img.img_bilateral)


# # In[10]:


# img.apply_gabor(kernel_bank= fil.get_chosen_gabor_bank())


# # In[11]:


# IMG.fromarray(img.img_gabor)


# # In[38]:


# img.create_skeleton(kernels=fil.get_chosen_skeleton_filter(),threshold=5)


# # In[39]:


# IMG.fromarray(255-img.img_skeleton)


# # In[40]:


# seed_node_list = lat.get_seed_node_list(img.img_skeleton)


# # In[41]:


# len(seed_node_list)


# # In[42]:


# lattice = Lattice(image_arr_2d=img.img_gabor)


# # In[44]:


# lattice.generate_lattice_graph(eight_connected=False)


# # In[21]:


# images =  [(0.7, img.img_gabor),(0.3, img.img_array)]


# # In[22]:


# lattice.assign_cost(images=images,alpha=5,log=False,override=True)


# # In[23]:


# import preprocess.algorithms.ature_mst as mst


# # In[24]:


# # seed_node_list = lat.get_seed_node_list(img.img_skeleton)
# # mst.run_dijkstra(lattice_object=lattice,seed=seed_node_list,number_of_seeds=5, weight_limit_per_seed=100000)


# # In[ ]:


# seed_node_list = lat.get_seed_node_list(img.img_skeleton)
# mst.run_mst(lattice_object=lattice,seed=seed_node_list,threshold=6)
# IMG.fromarray(lattice.accumulator)


# # In[ ]:


# seed_node_list = lat.get_seed_node_list(img.img_skeleton)
# mst.run_mst(lattice_object=lattice,seed=seed_node_list,threshold=6.5)
# IMG.fromarray(lattice.accumulator)


# # In[ ]:


# seed_node_list = lat.get_seed_node_list(img.img_skeleton)
# mst.run_mst(lattice_object=lattice,seed=seed_node_list,threshold=7)
# IMG.fromarray(lattice.accumulator)


# # In[ ]:


# seed_node_list = lat.get_seed_node_list(img.img_skeleton)
# mst.run_mst(lattice_object=lattice,seed=seed_node_list,threshold=7.5)
# IMG.fromarray(lattice.accumulator)


# # In[26]:


# IMG.fromarray(lattice.accumulator)

