import math
import os

import torch
from torch.utils.data.dataset import Dataset

import utils.img_utils as imgutil
from commons.IMAGE import Image
import numpy as np


class PatchesGenerator(Dataset):
    def __init__(self, Dirs=None, transform=None,
                 fget_mask=None, fget_truth=None, segment_mode=False, train_image_size=None):

        """
        :param Dirs: Should contain paths to directories images, mask, and truth by the same name.
        :param :train_image_size (w, h)
        :param transform:
        :param fget_mask: mask file getter
        :param fget_truth: ground truth file getter
        :param segment_mode: True only when running a model to segment since it returns the image id and
               pixel positions for each patches. DO NOT USE segment_mode while training or evaluating the model
        """

        self.transform = transform
        self.img_width, self.img_height = train_image_size
        self.IDs = []
        self.file_names = os.listdir(Dirs['images'])
        self.images = {}
        self.segment_mode = segment_mode
        for ID, img_file in enumerate(self.file_names):

            img_obj = Image()

            img_obj.load_file(data_dir=Dirs['images'], file_name=img_file)
            img_obj.working_arr = imgutil.whiten_image2d(img_obj.image_arr[:, :, 1])

            img_obj.load_mask(mask_dir=Dirs['mask'], fget_mask=fget_mask, erode=True)
            img_obj.load_ground_truth(gt_dir=Dirs['truth'], fget_ground_truth=fget_truth)

            for i in range(0, img_obj.working_arr.shape[0], 11):

                row_from, row_to = i, min(i + 11, img_obj.working_arr.shape[0])

                if abs(row_from - row_to) != 11:
                    row_from = img_obj.working_arr.shape[0] - 11
                    row_to = img_obj.working_arr.shape[0]

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

        # shuffle(self.IDs)
        print('### ' + str(self.__len__()) + ' patches found.')

    def __getitem__(self, index):
        ID, row_from, row_to = self.IDs[index]
        img_tensor = self.images[ID].working_arr[row_from:row_to, :][..., None]
        y = self.images[ID].ground_truth[row_from:row_to, :]
        y[y == 255] = 1
        y = torch.LongTensor(y)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        if self.segment_mode:
            return ID, np.array([row_from, row_to]), img_tensor, y

        return img_tensor, y

    def __len__(self):
        return len(self.IDs)
