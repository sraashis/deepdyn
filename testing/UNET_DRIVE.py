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
from neuralnet.unet.unet_dataloader import PatchesGenerator, PatchesGeneratorPerImgObj
from neuralnet.unet.model.unet import UNet

sep = os.sep
Dirs = {}
Dirs['checkpoint'] = 'assests' + sep + 'nnet_models'
Dirs['data'] = 'data' + sep + 'DRIVE' + sep + 'training'
Dirs['images'] = Dirs['data'] + sep + 'images'
Dirs['mask'] = Dirs['data'] + sep + 'mask'
Dirs['truth'] = Dirs['data'] + sep + '1st_manual'

TestDirs = {}
TestDirs['data'] = 'data' + sep + 'DRIVE' + sep + 'test'
TestDirs['images'] = TestDirs['data'] + sep + 'images'
TestDirs['mask'] = TestDirs['data'] + sep + 'mask'
TestDirs['truth'] = TestDirs['data'] + sep + '1st_manual'

ValidationDirs = {}
ValidationDirs['data'] = 'data' + sep + 'DRIVE' + sep + 'test'
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
    return file_name.split('_')[0] + '_manual1.gif'


def get_mask_file_test(file_name):
    return file_name.split('_')[0] + '_test_mask.gif'


num_channels = 1
classes = {'background': 0, 'vessel': 1, }
batch_size = 8
num_classes = len(classes)
epochs = 220
patch_rows, patch_cols = 388, 388  # height by width of training patches
use_gpu = True

#### Images to train/validate per epoch , None means USE ALL data####
train_size = None
validation_size = None
checkpoint_file = 'PytorchCheckpointUnetDRIVE.nn.tar'



transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# ### Load train data

# In[4]:


trainset = PatchesGenerator(Dirs=Dirs, train_image_size=(patch_rows, patch_cols),
                            transform=transform,
                            fget_mask=get_mask_file,
                            fget_truth=get_ground_truth_file, mode='train')

train_size = trainset.__len__() if train_size is None else train_size
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=0,
                                          sampler=WeightedRandomSampler(np.ones(trainset.__len__()), train_size))

# ### Load Validation Data

# In[5]:


validation_set = PatchesGenerator(Dirs=ValidationDirs, train_image_size=(patch_rows, patch_cols),
                                  transform=transform,
                                  fget_mask=get_mask_file_test,
                                  fget_truth=get_ground_truth_file, mode='train')

validation_size = validation_set.__len__() if validation_size is None else validation_size
validationloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                               shuffle=False, num_workers=0,
                                               sampler=WeightedRandomSampler(np.ones(validation_set.__len__()),
                                                                             validation_size, replacement=True))

# ### Define the network

# In[6]:


net = UNet(num_channels, num_classes)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# ### Train and evaluate network

# In[ ]:


# sys.stdout = open(checkpoint_file + 'LOG.txt', 'w')
trainer = UNetNNTrainer(model=net, checkpoint_dir=Dirs['checkpoint'], checkpoint_file=checkpoint_file, log_to_file=True)
# trainer.resume_from_checkpoint()
trainer.train(optimizer=optimizer, dataloader=trainloader, epochs=epochs, use_gpu=use_gpu,
              validationloader=validationloader, force_checkpoint=False, log_frequency=20)

# ### Test on a image

# In[22]:


img_obj = SegmentedImage()
img_obj.load_file(data_dir=TestDirs['images'], file_name='02_test.tif')
img_obj.load_mask(mask_dir=TestDirs['mask'], fget_mask=get_mask_file_test, erode=True)
img_obj.load_ground_truth(gt_dir=TestDirs['truth'], fget_ground_truth=get_ground_truth_file)

# In[31]:


transform_test = transforms.Compose([
    imgutil.whiten_image2d,
    transforms.ToPILImage(),
    transforms.ToTensor()
])

testset = PatchesGeneratorPerImgObj(img_obj=img_obj, train_image_size=(patch_rows, patch_cols),
                                    transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=testset.__len__(),
                                         shuffle=False, num_workers=0, sampler=None)

# In[32]:


scores, y_pred, y_true = trainer.evaluate(dataloader=testloader, use_gpu=use_gpu, force_checkpoint=False)
# mnt.plot_confusion_matrix(y_pred=y_pred, y_true=y_true, classes=classes)


# ### Merge the output to form a single image

# In[33]:

#
# import neuralnet.unet.utils as ut
#
# # In[34]:
#
#
# ppp = ut.merge_patches(scores, img_obj.working_arr.shape, (patch_rows, patch_cols))

# In[35]:


# IMG.fromarray(ppp)

# ### FAST MST algorithm

# In[ ]:


# params = {'sk_threshold': 150,
#           'alpha': 7.0,
#           'orig_contrib': 0.3,
#           'seg_threshold': 24}
#
# img_obj.working_arr = None  # todo
# img_obj.generate_skeleton(threshold=params['sk_threshold'])
# img_obj.generate_lattice_graph()

# In[12]:


for x, i in enumerate(testloader):
    break


