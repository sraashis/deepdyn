import math
import os

import torch
from torch.utils.data.dataset import Dataset

import utils.img_utils as imgutil
from commons.IMAGE import Image
import numpy as np
from random import shuffle
import random
import copy
import cv2


class PatchesGenerator(Dataset):
    def __init__(self, Dirs=None, transform=None,
                 fget_mask=None, fget_truth=None, train_image_size=None, pixel_offset=5,
                 mode=None):

        """
        :param Dirs: Should contain paths to directories images, mask, and truth by the same name.
        :param transform:
        :param fget_mask: mask file getter
        :param fget_truth: ground truth file getter
        :param pixel_offset: Offset pixels to increase the train size. Should be equal to the img_width while testing.
        :param mode: Takes value 'train' or 'eval'
        """

        self.transform = transform
        self.num_rows, self.num_cols = train_image_size
        self.train_images = []
        self.file_names = os.listdir(Dirs['images'])
        self.images = {}
        self.mode = mode
        for ID, img_file in enumerate(self.file_names):
            img_obj = Image()

            img_obj.load_file(data_dir=Dirs['images'], file_name=img_file)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_obj.working_arr = clahe.apply(img_obj.image_arr[:, :, 1])

            img_obj.load_mask(mask_dir=Dirs['mask'], fget_mask=fget_mask, erode=True)
            img_obj.load_ground_truth(gt_dir=Dirs['truth'], fget_ground_truth=fget_truth)

            self._initialize_keys(img_obj=img_obj, pixel_offset=pixel_offset, ID=str(ID), mode=mode)

            # if mode == 'train' and random.random() <= 0.10:
            #     img_obj1 = copy.deepcopy(img_obj)
            #     img_obj1.working_arr = img_obj.ground_truth.copy()
            #     self._initialize_keys_truth(img_obj=img_obj1, pixel_offset=pixel_offset, ID=str(ID) + '-reg')

            if mode == 'train':
                shuffle(self.train_images)

        print('### ' + str(self.__len__()) + ' patches found.')

    def _initialize_keys(self, img_obj=None, pixel_offset=None, ID=None, mode=None):
        for i in range(0, img_obj.working_arr.shape[0], pixel_offset):

            row_from, row_to = i, min(i + self.num_rows, img_obj.working_arr.shape[0])

            # Last patch could be of different size. So we adjust it to make consistent input size.
            if abs(row_from - row_to) != self.num_rows:
                row_from = img_obj.working_arr.shape[0] - self.num_rows
                row_to = img_obj.working_arr.shape[0]

            # only include patch that has at least one pixel in first row that is inside the mask.
            if mode == 'eval' or 255 in img_obj.ground_truth[row_from, :]:
                self.train_images.append([ID, row_from, row_to])
        self.images[ID] = img_obj

    def _initialize_keys_truth(self, img_obj=None, pixel_offset=None, ID=None, mode=None):
        for i in range(0, img_obj.working_arr.shape[0], pixel_offset):

            row_from, row_to = i, min(i + self.num_rows, img_obj.working_arr.shape[0])

            # Last patch could be of different size. So we adjust it to make consistent input size.
            if abs(row_from - row_to) != self.num_rows:
                row_from = img_obj.working_arr.shape[0] - self.num_rows
                row_to = img_obj.working_arr.shape[0]

            # only include patch that has at least one pixel in first row that is inside the mask.
            if 255 in img_obj.ground_truth[row_from, :] or mode == 'eval':
                self.train_images.append([ID, row_from, row_to])
        self.images[ID] = img_obj

    def __getitem__(self, index):
        ID, row_from, row_to = self.train_images[index]
        img_tensor = self.images[ID].working_arr[row_from:row_to, :][..., None]
        y = self.images[ID].ground_truth[row_from:row_to, :]
        y[y == 255] = 1
        y = torch.LongTensor(y)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        if self.mode == 'eval':
            return ID, np.array([row_from, row_to]), img_tensor, y

        return img_tensor, y

    def __len__(self):
        return len(self.train_images)
