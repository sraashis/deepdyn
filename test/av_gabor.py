
# coding: utf-8

# In[2]:


import os
import numpy as np


# In[3]:


os.chdir('D:\\idea projects\\pycharm projects\\ature\\')
from commons.IMAGE import Image
import preprocess.av.image_filters as fil
import cv2 as ocv


# In[13]:


img = Image('wide_image_03.mat')
img.apply_bilateral(img.img_array[:, :, 1])


# In[14]:


dif_bilateral = img.get_signed_diff_int8(img.img_array[:,:,1],img.img_bilateral)


# In[15]:


img.show_image(255-dif_bilateral)


# In[16]:


img.apply_gabor(255-dif_bilateral,filter_bank=fil.get_chosen_gabor_bank())
img.show_image(255-img.img_gabor)

