import math
import os

import torch
from torch.utils.data.dataset import Dataset

import utils.img_utils as imgutil
from commons.IMAGE import Image
import numpy as np
from random import shuffle


class PatchesGenerator(Dataset):
    def __init__(self, Dirs=None, transform=None,
                 fget_mask=None, fget_truth=None, patch_rows=None, pixel_offset=5,
                 mode=None):

        """
        :param Dirs: Should contain paths to directories images, mask, and truth by the same name.
        :param :train_image_size (w, h)
        :param transform:
        :param fget_mask: mask file getter
        :param fget_truth: ground truth file getter
        :param pixel_offset: Offset pixels to increase the train size. Should be equal to img_width while testing.
        :param mode: Takes value 'train' or 'test'
        """

        self.transform = transform
        self.patch_cols = patch_rows
        self.IDs = []
        self.file_names = os.listdir(Dirs['images'])
        self.images = {}
        self.mode = mode
        for ID, img_file in enumerate(self.file_names):
            img_obj = Image()

            img_obj.load_file(data_dir=Dirs['images'], file_name=img_file)
            img_obj.working_arr = imgutil.whiten_image2d(img_obj.image_arr[:, :, 1])

            img_obj.load_mask(mask_dir=Dirs['mask'], fget_mask=fget_mask, erode=True)
            img_obj.load_ground_truth(gt_dir=Dirs['truth'], fget_ground_truth=fget_truth)
#             img_obj.working_arr = img_obj.ground_truth.copy()

            self._initialize_keys(img_obj=img_obj, pixel_offset=pixel_offset, ID=str(ID))

        print('### ' + str(self.__len__()) + ' patches found.')

    def _initialize_keys(self, img_obj=None, pixel_offset=None, ID=None):
        for i in range(0, img_obj.working_arr.shape[0], pixel_offset):

            row_from, row_to = i, min(i + self.patch_cols, img_obj.working_arr.shape[1])

            if abs(row_from - row_to) != self.patch_cols:
                row_from = img_obj.working_arr.shape[0] - self.patch_cols
                row_to = img_obj.working_arr.shape[0]

            if 255 in img_obj.ground_truth[row_from, :]:
                self.IDs.append([ID, row_from, row_to])

        # Find the average of background pixels
        tot = 0.0
        c = 0
        for x in range(img_obj.working_arr.shape[0]):
            for y in range(img_obj.working_arr.shape[1]):
                if img_obj.mask[x, y] == 255 and img_obj.ground_truth[x, y] == 255:
                    tot += img_obj.working_arr[x, y]
                    c += 1

        img_obj.working_arr[img_obj.mask == 0] = math.ceil(tot / c)
        self.images[ID] = img_obj

    def __getitem__(self, index):
        ID, row_from, row_to = self.IDs[index]
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
        return len(self.IDs)
