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
        self.file_names = os.listdir(Dirs['images'])
        self.images = []
        self.mode = mode
        for ID, img_file in enumerate(self.file_names):
            img_obj = Image()

            img_obj.load_file(data_dir=Dirs['images'], file_name=img_file)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_obj.working_arr = clahe.apply(img_obj.image_arr[:, :, 1])

            img_obj.load_mask(mask_dir=Dirs['mask'], fget_mask=fget_mask, erode=True)
            img_obj.load_ground_truth(gt_dir=Dirs['truth'], fget_ground_truth=fget_truth)

            if mode == 'train':
                shuffle(self.images)
            self.images.append( img_obj)

        print('### ' + str(self.__len__()) + ' patches found.')

    def __getitem__(self, index):
        img_tensor = self.images[index].working_arr[..., None].copy()
        y = self.images[index].ground_truth.copy()
        y[y == 255] = 1
        y = torch.LongTensor(y)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        if self.mode == 'eval':
            return img_tensor, y

        return img_tensor, y

    def __len__(self):
        return len(self.images)
