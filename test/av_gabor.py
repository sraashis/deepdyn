
# coding: utf-8

# In[1]:


import os
import numpy as np


# In[2]:


os.chdir('D:\\idea projects\\pycharm projects\\ature\\')
from commons.IMAGE import Image
import preprocess.av.image_filters as fil
import cv2 as ocv


# In[3]:


import preprocess.av.av_utils as av


# In[4]:


img = Image('wide_image_03.mat')
img.apply_bilateral(img.img_array[:, :, 1])


# In[5]:


dif_bilateral = img.get_signed_diff_int8(img.img_array[:,:,1],img.img_bilateral)


# In[6]:
import networkx as nx
nx.number_of_nodes()

img.from_array(dif_bilateral)


# In[6]:


img.apply_gabor(255-dif_bilateral,filter_bank=fil.get_chosen_gabor_bank())
img.from_array(255-img.img_gabor)


# In[7]:


img.create_skeleton_by_threshold(array_2d=img.img_gabor,threshold=5)


# In[8]:


# img.img_skeleton[255-img.img_gabor]
img.from_array(img.img_skeleton)


# In[16]:


img.img_skeleton


# In[23]:


255-img.img_gabor


# In[11]:


img.from_array(255-img.img_skeleton)


# In[11]:


# img.from_array(255-img.img_skeleton)


# In[9]:


img.create_lattice_graph(image_arr_2d=255-img.img_gabor)


# In[10]:


images =  [(0.25, img.img_array[:,:,1]),(0.25, img.img_bilateral), (0.5, img.img_gabor)]


# In[11]:


img.assign_node_metrics(graph=img.lattice[0],metrics=img.img_skeleton)


# In[12]:


img.assign_cost(img.lattice[0], images=images, alpha=15)


# In[13]:


g = img.lattice[0]


# In[18]:


g[(69,712)]


# In[1]:


import networkx as nx


# In[1]:


# prims = nx.algorithms.prim_mst(g,weight='cost')


# In[15]:


g[(69,721)]


# In[2]:


gx = nx.Graph()


# In[3]:


gx.add_node(1)


# In[4]:


gx.add_node(2)


# In[5]:


gx.add_node(3)


# In[6]:


gx.add_node(4)


# In[7]:


gx.add_node(5)


# In[8]:


gx.add_node(6)


# In[9]:


gx.add_node(7)


# In[10]:


gx.add_node(8)


# In[11]:


gx.add_node(9)


# In[26]:


gx.add_edge(4,9,{'cost':4})


# In[16]:


import matplotlib.pyplot as plt


# In[27]:


nx.draw_networkx(gx)
plt.show()


# In[113]:


import heapq as hp


# In[117]:


import preprocess.av.av_utils as av


# In[32]:


import preprocess.algorithms.mst as mst


# In[38]:


res = mst.prim_mst(graph=gx,weight='cost')


# In[52]:


res[1]


# In[60]:


nx.draw_networkx(gx)
plt.show()


# In[ ]:


nodes = gx.nodes()


# In[ ]:


while nodes:
    x=gx.edges(u)
    print(x)
    for i,j in x:
        print(j)
    break


# In[112]:


from heapq import heappop, heappush
from itertools import *


# In[115]:


c = count()
a = [1.5,2.5,4.5,5.5,6.5,3.5,4.5,0.1,10.5,9.5,11.4,16.4,43.4,21.4,0.2,-1]
acc = []
for i in a:
    heappush(acc, heappush(acc,a))


# In[106]:



import itertools
from commons.timer import check_time
from numba import *
from multiprocessing import Array, Process


# In[110]:


test = np.zeros_like(img.img_skeleton)
all_nodes = g.nodes()


# In[98]:


@check_time
def get_skeleton1():
    nodes = all_nodes[0:300000]
    while nodes:
        n = nodes.pop(0)
        if g[n]['skeleton']==0:
            test[n[0],n[1]] = 255


# In[99]:


@check_time
def get_skeleton2():
    nodes = all_nodes[300000:600000]
    while nodes:
        n = nodes.pop(0)
        if g[n]['skeleton']==0:
            test[n[0],n[1]] = 255


# In[100]:


@check_time
def get_skeleton3():
    nodes = all_nodes[600000:900000]
    while nodes:
        n = nodes.pop(0)
        if g[n]['skeleton']==0:
            test[n[0],n[1]] = 255


# In[101]:


@check_time
def get_skeleton4():
    nodes = all_nodes[900000:1200000]
    while nodes:
        n = nodes.pop(0)
        if g[n]['skeleton']==0:
            test[n[0],n[1]] = 255


# In[103]:


@check_time
def get_skeleton5():
    nodes = all_nodes[1200000:1450000]
    while nodes:
        n = nodes.pop(0)
        if g[n]['skeleton']==0:
            test[n[0],n[1]] = 255


# In[105]:


from multiprocessing import Process


# In[113]:


pr1 = Process(target=get_skeleton1())
pr1.start()
pr1.join()

pr2 = Process(target=get_skeleton2())
pr2.start()
pr2.join()

pr3 = Process(target=get_skeleton3())
pr3.start()
pr3.join()

pr3 = Process(target=get_skeleton3())
pr3.start()
pr3.join()

pr4 = Process(target=get_skeleton4())
pr4.start()
pr4.join()

pr5 = Process(target=get_skeleton5())
pr5.start()
pr5.join()


# In[91]:


get_skeleton()


# In[64]:


lst


# In[80]:


for i,j in get_skeleton():
    test[i,j] = 255


# In[114]:


img.from_array(test)


# In[119]:


import networkx as nx
import matplotlib.pyplot as plt


# In[130]:


@check_time
def job1():
    return nx.algorithms.prim_mst(nx.subgraph(g, all_nodes[620000:720000]),weight='cost')


# In[ ]:


spann = job1()


# In[123]:


nx.draw_networkx(spann)
plt.show()


# In[124]:


axr= np.zeros_like(img.img_skeleton)


# In[128]:


for i,j in spann.nodes():
    axr[i,j]=255
img.from_array(axr)

