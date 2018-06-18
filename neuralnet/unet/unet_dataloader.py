import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import utils.img_utils as imgutils
from commons.IMAGE import Image


class PatchesGenerator(Dataset):
    def __init__(self, Dirs=None, transform=None,
                 fget_mask=None, fget_truth=None, train_image_size=(388, 388), pad_row_col=[(92, 92), (92, 92)]):

        """
        :param Dirs: Should contain paths to directories images, mask, and truth by the same name.
        :param transform:
        :param fget_mask: mask file getter
        :param fget_truth: ground truth file getter
        :param mode: Takes value 'train' or 'eval'
        """

        self.transform = transform
        self.patches_indexes = []
        self.images = {}
        self.train_image_size = train_image_size
        self.pad_row_col = pad_row_col
        self.Dirs = Dirs
        self.fget_mask = fget_mask
        self.fget_truth = fget_truth
        files = os.listdir(Dirs['images']) if Dirs is not None else []
        for ID, img_file in enumerate(files):
            img_obj = self._get_image_obj(img_file)
            for chunk_ix in imgutils.get_chunk_indexes(img_obj.working_arr.shape, self.train_image_size):
                self.patches_indexes.append([ID] + chunk_ix)
            self.images[ID] = img_obj

        random.shuffle(self.patches_indexes)
        print('### ' + str(self.__len__()) + ' patches found.')

    def _get_image_obj(self, img_file=None):
        img_obj = Image()
        img_obj.load_file(data_dir=self.Dirs['images'], file_name=img_file)
        img_obj.load_mask(mask_dir=self.Dirs['mask'], fget_mask=self.fget_mask, erode=True)
        img_obj.load_ground_truth(gt_dir=self.Dirs['truth'], fget_ground_truth=self.fget_truth)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_obj.working_arr = clahe.apply(img_obj.image_arr[:, :, 1])
        if img_obj.mask is not None:
            x = np.logical_and(True, img_obj.mask == 255)
            img_obj.working_arr[img_obj.mask == 0] = img_obj.working_arr[x].mean()
        return img_obj

    def __getitem__(self, index):

        ID, row_from, row_to, col_from, col_to = self.patches_indexes[index]
        img_tensor = self.images[ID].working_arr[row_from:row_to, col_from:col_to]
        img_tensor = np.pad(img_tensor, self.pad_row_col, 'reflect')
        y = self.images[ID].ground_truth[row_from:row_to, col_from:col_to]

        img_tensor = img_tensor[..., None]
        y[y == 255] = 1
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.LongTensor(y)

    def __len__(self):
        return len(self.patches_indexes)


class PatchesGeneratorPerImgObj(PatchesGenerator):
    def __init__(self, img_obj=None, train_image_size=(388, 388), pad_row_col=[(92, 92), (92, 92)], transform=None):
        super().__init__(transform=transform, train_image_size=train_image_size, pad_row_col=pad_row_col)
        self.images[0] = img_obj
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_obj.working_arr = clahe.apply(img_obj.image_arr[:, :, 1])
        for chunk_ix in imgutils.get_chunk_indexes(img_obj.working_arr.shape, self.train_image_size):
            self.patches_indexes.append([0] + chunk_ix)
        print('### ' + str(self.__len__()) + ' patches found.')
