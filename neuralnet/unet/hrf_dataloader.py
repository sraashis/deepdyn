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
                 fget_mask=None, fget_truth=None, train_image_size=(388, 388),
                 mode=None):

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
        self.mode = mode
        self.patch_rows, self.patch_cols = train_image_size
        self.file_names = os.listdir(Dirs['images']) if Dirs is not None else []

        for ID, img_file in enumerate(self.file_names):
            img_obj = Image()

            img_obj.load_file(data_dir=Dirs['images'], file_name=img_file)

            img_obj.load_mask(mask_dir=Dirs['mask'], fget_mask=fget_mask, erode=True)
            img_obj.load_ground_truth(gt_dir=Dirs['truth'], fget_ground_truth=fget_truth)

            # Contrast equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_obj.working_arr = clahe.apply(img_obj.image_arr[:, :, 1])

            for chunk_ix in imgutils.get_chunk_indexes(img_obj.working_arr.shape, (self.patch_rows, self.patch_cols)):
                self.patches_indexes.append([ID] + chunk_ix)

            if mode == 'train':
                random.shuffle(self.patches_indexes)
            self.images[ID] = img_obj

        print('### ' + str(self.__len__()) + ' patches found.')

    def __getitem__(self, index):

        ID, row_from, row_to, col_from, col_to = self.patches_indexes[index]
        img_tensor = self.images[ID].working_arr[row_from:row_to, col_from:col_to]
        img_tensor = np.pad(img_tensor, [92], 'reflect')
        y = self.images[ID].ground_truth[row_from:row_to, col_from:col_to]

        img_tensor = img_tensor[..., None]
        y[y == 255] = 1

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.LongTensor(y)

    def __len__(self):
        return len(self.patches_indexes)


class PatchesGeneratorPerImgObj(PatchesGenerator):
    def __init__(self, img_obj=None, train_image_size=(388, 388), transform=None, mode='eval'):
        super().__init__(transform=transform, train_image_size=train_image_size, mode=mode)
        self.images[0] = img_obj
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_obj.working_arr = clahe.apply(img_obj.image_arr[:, :, 1])
        for chunk_ix in imgutils.get_chunk_indexes(img_obj.working_arr.shape, (self.patch_rows, self.patch_cols)):
            self.patches_indexes.append([0] + chunk_ix)
        print('### ' + str(self.__len__()) + ' patches found.')
