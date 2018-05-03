# coding: utf-8

# ### Initialize notebook

# In[ ]:


# !/home/akhanal1/Spring2018/pl-env/bin/python3.5
# Torch imports

import os
import sys

sys.path.append('/home/akhanal1/Spring2018/ature')
os.chdir('/home/akhanal1/Spring2018/ature')

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import PIL.Image as IMG
from torch.utils.data.sampler import WeightedRandomSampler

import numpy as np

from neuralnet.utils.torch_dataset import TorchPatchesGenerator
from utils import img_utils as imgutil
from neuralnet.torchtrainer import NNTrainer

# %load_ext autoreload
# %autoreload 2


# In[ ]:


### Define folders. Create if needed.
sep = os.sep
Dirs = {}
Dirs['checkpoint'] = 'assests' + sep + 'nnet_models'
Dirs['data'] = 'data' + sep + 'DRIVE' + sep + 'training'
Dirs['images'] = Dirs['data'] + sep + 'images'
Dirs['mask'] = Dirs['data'] + sep + 'mask'
Dirs['truth'] = Dirs['data'] + sep + '1st_manual'

TestDirs = {}
TestDirs['data'] = 'data' + sep + 'DRIVE' + sep + 'test'
TestDirs['images'] = TestDirs['data'] + sep + 'test_images'
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


classes = {'background': 0, 'vessel': 1, }
batch_size = 52
num_classes = len(classes)
epochs = 5
patch_size = 51
use_gpu = False

#### Images to train/validate per epoch ####
train_size = 500000
validation_size = 100000
checkpoint_file = 'PytorchCheckpoint51.nn.tar'


# ### Define the network

# In[ ]:


class Net(nn.Module):
    def __init__(self, width, channels):
        super(Net, self).__init__()

        self.channels = channels
        self.width = width

        self.kern_size = 11
        self.kern_stride = 2
        self.kern_padding = 1
        self.mxp_kern_size = 1
        self.mxp_stride = 1
        self.pool1 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv1 = nn.Conv2d(self.channels, 24, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.kern_size = 5
        self.kern_stride = 1
        self.kern_padding = 3
        self.mxp_kern_size = 2
        self.mxp_stride = 2
        self.pool2 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv2 = nn.Conv2d(24, 52, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.kern_size = 5
        self.kern_stride = 1
        self.kern_padding = 1
        self.mxp_kern_size = 2
        self.mxp_stride = 2
        self.pool3 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv3 = nn.Conv2d(52, 96, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.kern_size = 3
        self.kern_stride = 2
        self.kern_padding = 1
        self.mxp_kern_size = 1
        self.mxp_stride = 1
        self.pool4 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv4 = nn.Conv2d(96, 30, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.linearWidth = 30 * int(self.width) * int(self.width)
        self.fc1 = nn.Linear(self.linearWidth, 512)
        self.fc2 = nn.Linear(512, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(-1, self.linearWidth)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def _update_output_size(self):
        temp = self.width
        self.width = ((self.width - self.kern_size + 2 * self.kern_padding) / self.kern_stride) + 1
        temp1 = self.width
        self.width = ((self.width - self.mxp_kern_size) / self.mxp_stride) + 1
        print('Output width[ ' + str(temp) + ' -conv-> ' + str(temp1) + ' -maxpool-> ' + str(self.width) + ' ]')


width = patch_size
channels = 1
net = Net(width, channels)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# ### Load train data

# In[ ]:


transform = transforms.Compose([
    imgutil.whiten_image2d,
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(40),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

trainset = TorchPatchesGenerator(Dirs=Dirs, patch_size=patch_size,
                                 transform=transform,
                                 fget_mask=get_mask_file,
                                 fget_truth=get_ground_truth_file)

### Fix skewed classes by sampling based on class weights
labels = np.array(trainset.IDs)[:, 3]
_, ccounts_train = np.unique(labels, return_counts=True)
cweights_train = 1.0 / ccounts_train
dweights_train = np.array([cweights_train[t] for t in labels])

train_size = trainset.__len__() if train_size is None else train_size
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=3,
                                          sampler=WeightedRandomSampler(dweights_train, train_size))

# ### Load Validation Data

# In[ ]:


transform_val = transforms.Compose([
    imgutil.whiten_image2d,
    transforms.ToPILImage(),
    transforms.ToTensor()
])

validation_set = TorchPatchesGenerator(Dirs=ValidationDirs, patch_size=patch_size,
                                       transform=transform_val,
                                       fget_mask=get_mask_file_test,
                                       fget_truth=get_ground_truth_file)

labels_val = np.array(validation_set.IDs)[:, 3]
_, ccounts_val = np.unique(labels_val, return_counts=True)
cweights_val = 1.0 / ccounts_val
dweights_val = np.array([cweights_val[t] for t in labels_val])

validation_size = validation_set.__len__() if validation_size is None else validation_size
validationloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                               shuffle=False, num_workers=3,
                                               sampler=WeightedRandomSampler(dweights_val,
                                                                             validation_size, replacement=True))

# ### Train and evaluate the Network

# In[ ]:


trainer = NNTrainer(model=net, checkpoint_dir=Dirs['checkpoint'], checkpoint_file=checkpoint_file)
# trainer.resume_from_checkpoint()
trainer.train(optimizer=optimizer, dataloader=trainloader, epochs=epochs, use_gpu=use_gpu,
              validationloader=validationloader)
# trainer.resume_from_checkpoint()


# ### Test on a image

# In[ ]:


transform_test = transforms.Compose([
    imgutil.whiten_image2d,
    transforms.ToPILImage(),
    transforms.ToTensor()
])

testset = TorchPatchesGenerator(Dirs=TestDirs, patch_size=patch_size,
                                transform=transform_test,
                                fget_mask=get_mask_file_test,
                                fget_truth=get_ground_truth_file,
                                segment_mode=True)

sampler = WeightedRandomSampler(np.ones(testset.__len__()), 5000, replacement=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=3, sampler=None)

# In[ ]:


IDs, IJs, scores, y_pred, y_true = trainer.evaluate(dataloader=testloader, use_gpu=use_gpu, force_checkpoint=False)
# mnt.plot_confusion_matrix(y_pred=y_pred, y_true=y_true, classes=classes)


# ### Convolve throughout the image to generate segmented image based on trained Network

# In[ ]:


sc = np.exp(scores.copy())
seg = np.zeros(testset.images[0].working_arr.shape)
for val in zip(IDs, IJs, sc):
    image_id, (i, j), (b_prob, v_prob) = val
    seg[i, j] = 255 * v_prob

# In[ ]:


IMG.fromarray(seg.astype(np.uint8)).save(checkpoint_file + testset.images[0].file_name + '.png')

# In[ ]:


# params = {'sk_threshold': 150,
#           'alpha': 7.0,
#           'orig_contrib': 0.3,
#           'seg_threshold': 24}

# img_obj = SegmentedImage()

# img_obj.load_file(data_dir=TestDirs['images'], file_name=testset.images[0].file_name)
# img_obj.res['orig'] = img_obj.image_arr[:, :, 1]
# # img_obj.working_arr = 255 - seg.astype(np.uint8)

# img_obj.load_mask(mask_dir=TestDirs['mask'], fget_mask=get_mask_file_test, erode=True)
# img_obj.load_ground_truth(gt_dir=TestDirs['truth'], fget_ground_truth=get_ground_truth_file)

# img_obj.generate_skeleton(threshold=params['sk_threshold'])
# img_obj.generate_lattice_graph()


# In[ ]:


# tester = AtureTest(out_dir='')
# tester.run(params=params, save_images=False, img_obj=img_obj)
# img_obj.res['scores']
