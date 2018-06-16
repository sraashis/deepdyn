# !/home/akhanal1/miniconda3//bin/python3.5
# Torch imports
import os
import sys

sys.path.append('/home/akhanal1/ature')
os.chdir('/home/akhanal1/ature')

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import PIL.Image as IMG
from torch.utils.data.sampler import WeightedRandomSampler

import numpy as np

from commons.IMAGE import SegmentedImage

from utils import img_utils as imgutil
from neuralnet.unet.unet_trainer import UNetNNTrainer
from neuralnet.unet.unet_dataloader import PatchesGeneratorAV, PatchesGeneratorPerImgObj
from neuralnet.unet.model.unet import UNet

# In[2]:


### Define folders. Create if needed.
sep = os.sep
Dirs = {}
Dirs['checkpoint'] = 'assests' + sep + 'nnet_models'
Dirs['data'] = 'data' + sep + 'AV-WIDE' + sep + 'training'
Dirs['images'] = Dirs['data'] + sep + 'images'
Dirs['mask'] = Dirs['data'] + sep + 'mask'
Dirs['truth'] = Dirs['data'] + sep + '1st_manual'

TestDirs = {}
TestDirs['data'] = 'data' + sep + 'AV-WIDE' + sep + 'testing'
TestDirs['images'] = TestDirs['data'] + sep + 'images'
TestDirs['mask'] = TestDirs['data'] + sep + 'mask'
TestDirs['truth'] = TestDirs['data'] + sep + '1st_manual'

ValidationDirs = {}
ValidationDirs['data'] = 'data' + sep + 'AV-WIDE' + sep + 'testing'
ValidationDirs['images'] = ValidationDirs['data'] + sep + 'validation_images'
ValidationDirs['mask'] = ValidationDirs['data'] + sep + 'mask'
ValidationDirs['truth'] = ValidationDirs['data'] + sep + '1st_manual'

for k, folder in Dirs.items():
    os.makedirs(folder, exist_ok=True)
for k, folder in TestDirs.items():
    os.makedirs(folder, exist_ok=True)
for k, folder in ValidationDirs.items():
    os.makedirs(folder, exist_ok=True)


def get_mask_file(file_name):
    return file_name.split('_')[0] + '_training_mask.gif'


def get_ground_truth_file(file_name):
    return file_name.split('.')[0] + '_vessels.png'


num_channels = 1
classes = {'background': 0, 'vessel': 1, }
batch_size = 3
num_classes = len(classes)
epochs = 120
patch_rows, patch_cols = 388, 388  # height by width of training patches
use_gpu = False

#### Images to train/validate per epoch , None means USE ALL data####
train_size = None
validation_size = None
checkpoint_file = 'PytorchCheckpointUnetAV-WIDE.nn.tar'

# ### Transformations

# In[3]:


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# ### Load train data

# In[ ]:


trainset = PatchesGeneratorAV(Dirs=Dirs, train_image_size=(patch_rows, patch_cols),
                              transform=transform,
                              fget_mask=get_mask_file,
                              fget_truth=get_ground_truth_file, mode='train')

train_size = trainset.__len__() if train_size is None else train_size
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=0,
                                          sampler=WeightedRandomSampler(np.ones(trainset.__len__()), train_size))

# ### Load Validation Data

# In[66]:


validation_set = PatchesGeneratorAV(Dirs=ValidationDirs, train_image_size=(patch_rows, patch_cols),
                                  transform=transform,
                                  fget_mask=get_mask_file,
                                  fget_truth=get_ground_truth_file, mode='train')

validation_size = validation_set.__len__() if validation_size is None else validation_size
validationloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                               shuffle=False, num_workers=0,
                                               sampler=WeightedRandomSampler(np.ones(validation_set.__len__()),
                                                                             validation_size, replacement=True))

# ### Define the network

# In[5]:


net = UNet(num_channels, num_classes)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# ### Train and evaluate network

# In[6]:


trainer = UNetNNTrainer(model=net, checkpoint_dir=Dirs['checkpoint'], checkpoint_file=checkpoint_file)
trainer.resume_from_checkpoint()
# trainer.train(optimizer=optimizer, dataloader=trainloader, epochs=4, use_gpu=use_gpu, 
#               validationloader=validationloader, force_checkpoint=False, log_frequency=20)


# ### Test on a image

# In[7]:


img_obj = SegmentedImage()
img_obj.load_file(data_dir=TestDirs['images'], file_name='wide_image_16.png')
img_obj.load_ground_truth(gt_dir=TestDirs['truth'], fget_ground_truth=get_ground_truth_file)

# In[8]:


transform_test = transforms.Compose([
    imgutil.whiten_image2d,
    transforms.ToPILImage(),
    transforms.ToTensor()
])

testset = PatchesGeneratorPerImgObj(img_obj=img_obj, train_image_size=(patch_rows, patch_cols),
                                    transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0, sampler=None)

# In[9]:


# scores, y_pred, y_true = trainer.evaluate(dataloader=testloader, use_gpu=use_gpu, force_checkpoint=False)
# # mnt.plot_confusion_matrix(y_pred=y_pred, y_true=y_true, classes=classes)
#
#
# # ### Merge the output to form a single image
#
# # In[10]:
#
#
# import neuralnet.unet.utils as ut
#
# # In[11]:
#
#
# ppp = ut.merge_patches(scores, img_obj.image_arr[:, :, 1].shape, (patch_rows, patch_cols))
#
# # In[12]:
#
#
# # IMG.fromarray(ppp)
# img_obj.working_arr.shape
#
# # In[13]:
#
#
# IMG.fromarray(ppp)
#
# # In[22]:



