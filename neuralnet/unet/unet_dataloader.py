import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import neuralnet.unet.utils as unet_utils
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
        self.images = []
        self.labels = []
        self.mode = mode
        self.patch_rows, self.patch_cols = train_image_size
        self.file_names = os.listdir(Dirs['images']) if Dirs is not None else []

        for ID, img_file in enumerate(self.file_names):
            img_obj = Image()

            img_obj.load_file(data_dir=Dirs['images'], file_name=img_file)

            img_obj.load_mask(mask_dir=Dirs['mask'], fget_mask=fget_mask, erode=True)
            img_obj.load_ground_truth(gt_dir=Dirs['truth'], fget_ground_truth=fget_truth)

            self.load_patches(img_obj)

            if mode == 'train':
                combined = list(zip(self.images, self.labels))
                random.shuffle(combined)
                self.images[:], self.labels[:] = zip(*combined)

        print('### ' + str(self.__len__()) + ' patches found.')

    def __getitem__(self, index):
        img_tensor = np.pad(self.images[index], [92], 'reflect')
        y = self.labels[index]

        img_tensor = img_tensor[..., None]
        y[y == 255] = 1

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.LongTensor(y)

    def __len__(self):
        return len(self.images)

    def load_patches(self, img_obj):

        # Contrast equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_obj.working_arr = clahe.apply(img_obj.image_arr[:, :, 1])

        # Initialize region outside mask with average value.
        x = np.logical_and(True, img_obj.mask == 255)
        img_obj.working_arr[img_obj.mask == 0] = img_obj.working_arr[x].mean()
        img_rows, img_cols = img_obj.working_arr.shape
        rows_offset = 0
        for i in range(0, img_rows, self.patch_rows):
            row_from = i - rows_offset
            row_to = i - rows_offset + self.patch_rows
            if row_to > img_rows:
                row_to = img_rows
                row_from = img_rows - self.patch_rows
            rows_offset = unet_utils.equalize_offset(img_rows, self.patch_rows)

            cols_offset = 0
            for j in range(0, img_cols, self.patch_cols):
                col_from = j - cols_offset
                col_to = j - cols_offset + self.patch_cols
                if col_to > img_cols:
                    col_to = img_cols
                    col_from = img_cols - self.patch_cols
                cols_offset = unet_utils.equalize_offset(img_cols, self.patch_cols)

                self.images.append(img_obj.working_arr[row_from:row_to, col_from:col_to])
                self.labels.append(img_obj.ground_truth[row_from:row_to, col_from:col_to])


class PatchesGeneratorPerImgObj(PatchesGenerator):
    def __init__(self, img_obj=None, train_image_size=(388, 388), transform=None, mode='eval'):
        super().__init__(transform=transform, train_image_size=train_image_size, mode=mode)
        self.load_patches(img_obj)
        print('### ' + str(self.__len__()) + ' patches found.')
