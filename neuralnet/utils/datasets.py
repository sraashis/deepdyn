import itertools
import math
import os

import PIL.Image as IMG
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils import img_utils as imgutil
import neuralnet.utils.data_utils as datautils


class DriveDatasetFromFile4Way(Dataset):

    # data_path should contain pickled numpy array of dimension N * (D+1)
    # Where extra one dimension stores the correct lable among (0, 1, 2, 3)

    def __init__(self, data_path=None, height=None, width=None, transform=None):

        self.height = height
        self.width = width
        self.transform = transform

        self.data = None
        for data_file in os.listdir(data_path):

            data_file = os.path.join(data_path, data_file)
            print('Data file: ' + data_file)
            if self.data is None:
                self.data = np.load(data_file)
            else:
                self.data = np.concatenate((self.data, np.load(data_file)), axis=0)

        self.labels = torch.from_numpy(self.data[:, self.height * self.width])
        self.data = self.data[:, 0:self.height * self.width]

    def __getitem__(self, index):

        img_arr = self.data[index].reshape(self.height, self.width)
        img = IMG.fromarray(imgutil.whiten_image2d(img_arr))

        if self.transform is not None:
            img_tensor = self.transform(img)

        return img_tensor, self.labels[index]

    def __len__(self):
        return self.data.shape[0]


class DriveDatasetFromFile2Way(Dataset):

    # data_path should contain pickled numpy array of dimension N * (D+1)
    # Where extra one dimension stores the correct lable among (0, 1, 2, 3)

    def __init__(self, data_path=None, height=None, width=None, transform=None):

        self.height = height
        self.width = width
        self.transform = transform

        self.data = None
        for data_file in os.listdir(data_path):

            data_file = os.path.join(data_path, data_file)
            print('Data file: ' + data_file)
            if self.data is None:
                self.data = np.load(data_file)
            else:
                self.data = np.concatenate((self.data, np.load(data_file)), axis=0)

        self.labels = self.data[:, self.height * self.width]

        self.labels[self.labels == 0] = 1  # White (TP)
        self.labels[self.labels == 1] = 0  # Green (FP)
        self.labels[self.labels == 2] = 0  # Black (TN)
        self.labels[self.labels == 3] = 1  # Red (FN)

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


class DriveDatasetFromImageObj2Way(Dataset):

    def __init__(self, img_obj=None, patch_size=None, transform=None):

        self.img_obj = img_obj
        self.transform = transform
        self.patch_size = patch_size
        self.k_half = math.floor(patch_size / 2)

        self.w, self.h = img_obj.working_array.shape

        self.data = []
        for i, j in itertools.product(np.arange(self.w), np.arange(self.h)):
            if img_obj.mask[i, j] == 255:
                self.data.append((i, j))
        self.data = np.array(self.data)

        self.labels = np.zeros(self.data.shape[0])
        for ix, ij in enumerate(self.data, 0):
            i, j = ij
            if self.img_obj.ground_truth[i, j] == 255:
                self.labels[ix] = 1

        self.labels = torch.from_numpy(self.labels)

    def __getitem__(self, index):
        i = self.data[index][0]
        j = self.data[index][1]

        patch = np.full((self.patch_size, self.patch_size), 0, dtype=np.uint8)
        for k in range(-self.k_half, self.k_half + 1, 1):
            for l in range(-self.k_half, self.k_half + 1, 1):
                patch_i = i + k
                patch_j = j + l
                if self.img_obj.working_array.shape[0] > patch_i >= 0 and self.img_obj.working_array.shape[
                    1] > patch_j >= 0:
                    patch[self.k_half + k, self.k_half + l] = self.img_obj.working_array[patch_i, patch_j]

        img = IMG.fromarray(imgutil.whiten_image2d(patch))

        if self.transform is not None:
            img_tensor = self.transform(img)

        return i, j, img_tensor, self.labels[index]

    def __len__(self):
        return self.data.shape[0]


class DriveDatasetFromImageObj4Way(Dataset):

    def __init__(self, img_obj=None, patch_size=None, transform=None):

        self.img_obj = img_obj
        self.transform = transform
        self.patch_size = patch_size
        self.k_half = math.floor(patch_size / 2)

        self.w, self.h = img_obj.working_array.shape

        self.data = []
        for i, j in itertools.product(np.arange(self.w), np.arange(self.h)):
            if img_obj.mask[i, j] == 255:
                self.data.append((i, j))
        self.data = np.array(self.data)

        self.labels = np.zeros(self.data.shape[0])
        for ix, ij in enumerate(self.data, 0):
            i, j = ij
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
                if self.img_obj.working_array.shape[0] > patch_i >= 0 and self.img_obj.working_array.shape[
                    1] > patch_j >= 0:
                    patch[self.k_half + k, self.k_half + l] = self.img_obj.working_array[patch_i, patch_j]

        img = IMG.fromarray(imgutil.whiten_image2d(patch))

        if self.transform is not None:
            img_tensor = self.transform(img)

        return i, j, img_tensor, self.labels[index]

    def __len__(self):
        return self.data.shape[0]