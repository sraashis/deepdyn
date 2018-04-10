import itertools
import math
import os
from random import shuffle

import PIL.Image as IMG
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import neuralnet.utils.data_utils as datautils
import utils.img_utils as imgutil
from commons.IMAGE import Image


class DriveDatasetFromFile(Dataset):
    # data_path should contain pickled numpy array of dimension N * (D+1)
    # Where extra one dimension stores the correct lable among (0, 1, 2, 3)

    def __init__(self, data_path=None, height=None, width=None, num_classes=None, transform=None):

        self.height = height
        self.width = width
        self.transform = transform
        self.num_classes = num_classes

        self.data = None
        for data_file in os.listdir(data_path):
            try:
                data_file = os.path.join(data_path, data_file)
                if self.data is None:
                    self.data = np.load(data_file)
                else:
                    self.data = np.concatenate((self.data, np.load(data_file)), axis=0)
                print('Data file loaded: ' + data_file)
            except Exception as e:
                print('ERROR loading ' + data_file + ' : ' + str(e))
                continue

        self.labels = self.data[:, self.height * self.width]

        if num_classes == 2:
            for i, y in enumerate(self.labels):
                if y == 0 or y == 3:
                    self.labels[i] = 1
                elif y == 1 or y == 2:
                    self.labels[i] = 0

        self.labels = torch.from_numpy(self.labels)
        self.data = self.data[:, 0:self.height * self.width]

    def __getitem__(self, index):

        img_arr = self.data[index].reshape(self.height, self.width)
        img = IMG.fromarray(imgutil.whiten_image2d(img_arr))

        if self.transform is not None:
            img_tensor = self.transform(img)

        return img_tensor, self.labels[index]

    def __len__(self):
        return self.data.shape[0]


class DriveDatasetFromImageObj(Dataset):
    # Loads dataset directly from the image object in commons package.
    # returns "i, j, dataset, label" of the patch for further use during __next_item__ iteration

    def __init__(self, img_obj=None, patch_size=None, num_classes=None, transform=None):

        self.img_obj = img_obj
        self.transform = transform
        self.patch_size = patch_size
        self.k_half = math.floor(patch_size / 2)
        self.num_classes = num_classes

        self.w, self.h = img_obj.working_arr.shape

        self.data = []
        for i, j in itertools.product(np.arange(self.w), np.arange(self.h)):
            if img_obj.mask[i, j] == 255:
                self.data.append((i, j))

        self.data = np.array(self.data)
        self.labels = np.zeros(self.data.shape[0], dtype=np.uint8)
        for ix, ij in enumerate(self.data, 0):

            i, j = ij
            if self.num_classes == 2:
                if self.img_obj.ground_truth[i, j] == 255:
                    self.labels[ix] = 1
                else:
                    self.labels[ix] = 0

            elif self.num_classes == 4:
                self.labels[ix] = datautils.get_lable(i, j, self.img_obj.res['segmented'], self.img_obj.ground_truth)

        self.labels = torch.from_numpy(self.labels)

    def __getitem__(self, index):
        i = self.data[index][0]
        j = self.data[index][1]

        patch = np.full((self.patch_size, self.patch_size), 0, dtype=np.uint8)
        for k in range(-self.k_half, self.k_half + 1, 1):
            for l in range(-self.k_half, self.k_half + 1, 1):
                patch_i = i + k
                patch_j = j + l
                if self.img_obj.working_arr.shape[0] > patch_i >= 0 and self.img_obj.working_arr.shape[
                    1] > patch_j >= 0:
                    patch[self.k_half + k, self.k_half + l] = self.img_obj.working_arr[patch_i, patch_j]

        img = IMG.fromarray(imgutil.whiten_image2d(patch))

        if self.transform is not None:
            img_tensor = self.transform(img)

        return i, j, img_tensor, self.labels[index]

    def __len__(self):
        return self.data.shape[0]


class PatchesGenerator(Dataset):

    def __init__(self, Dirs=None, patch_size=None, num_classes=None, transform=None,
                 fget_mask=None, fget_ground=None, fget_segmented=None):
        self.data = []
        self.file_names = os.listdir(Dirs['image'])
        self.images = {}
        for ID, img_file in enumerate(self.file_names):

            img_obj = Image()

            img_obj.load_file(data_dir=Dirs['images'], file_name=img_file)
            img_obj.working_arr = imgutil.whiten_image2d(img_obj.working_arr)

            img_obj.load_mask(mask_dir=Dirs['mask'], fget_mask=fget_mask, erode=True)
            img_obj.load_ground_truth(gt_dir=Dirs['truth'], fget_ground_truth=fget_ground)

            segmented_file = os.path.join(Dirs['segmented'], fget_segmented(img_file))
            img_obj.res['segmented'] = imgutil.get_image_as_array(segmented_file, channels=1)

            for i, j in itertools.product(np.arange(self.w), np.arange(self.h)):
                if img_obj.mask[i, j] == 255:
                    self.data.append([ID, i, j])

            self.images[ID] = img_obj

        shuffle(self.data)
        self.labels = np.zeros(self.data.shape[0], dtype=np.uint8)

        for i, IDXY in enumerate(self.data, 0):
            ID, x, y = IDXY
            if self.num_classes == 2:
                if self.images[ID].ground_truth[x, y] == 255:
                    self.labels[i] = 1
                else:
                    self.labels[i] = 0

            elif self.num_classes == 4:
                self.labels[i] = datautils.get_lable(x, y, self.images[ID].res['segmented'],
                                                     self.images[ID].ground_truth)

        self.labels = torch.from_numpy(self.labels)

        self.transform = transform
        self.patch_size = patch_size
        self.num_classes = num_classes

    def __getitem__(self, index):
        ID, i, j = self.data[index]
        k_half = int(math.floor(self.patch_size / 2))
        patch = np.full((self.patch_size, self.patch_size), 0, dtype=np.uint8)
        for k in range(k_half, k_half + 1, 1):
            for l in range(-k_half, k_half + 1, 1):
                patch_i = i + k
                patch_j = j + l
                if self.images[ID].working_arr.shape[0] > patch_i >= 0 and self.images[ID].working_arr.shape[
                    1] > patch_j >= 0:
                    patch[self.k_half + k, self.k_half + l] = self.images[ID].working_arr[patch_i, patch_j]

        img_tensor = patch[..., None]
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return i, j, img_tensor, self.labels[index]

    def __len__(self):
        return len(self.data)
