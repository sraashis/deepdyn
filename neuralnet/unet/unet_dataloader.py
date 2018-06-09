import os
import random

import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
import torch

from commons.IMAGE import Image


class PatchesGenerator(Dataset):
    def __init__(self, Dirs=None, transform=None,
                 fget_mask=None, fget_truth=None, train_image_size=None,
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
        self.labels = []
        self.mode = mode
        for ID, img_file in enumerate(self.file_names):
            img_obj = Image()

            img_obj.load_file(data_dir=Dirs['images'], file_name=img_file)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_obj.working_arr = clahe.apply(img_obj.image_arr[:, :, 1])

            img_obj.load_mask(mask_dir=Dirs['mask'], fget_mask=fget_mask, erode=True)
            img_obj.load_ground_truth(gt_dir=Dirs['truth'], fget_ground_truth=fget_truth)

            x = np.logical_and(True, img_obj.mask == 255)
            img_obj.working_arr[img_obj.mask == 0] = img_obj.working_arr[x].mean()

            self.images.append(img_obj.working_arr[0:388, 0:388])
            self.images.append(img_obj.working_arr[0:388, 176:564])
            self.images.append(img_obj.working_arr[176:564, 0:388])
            self.images.append(img_obj.working_arr[176:564, 176:564])

            self.labels.append(img_obj.ground_truth[0:388, 0:388])
            self.labels.append(img_obj.ground_truth[0:388, 176:564])
            self.labels.append(img_obj.ground_truth[176:564, 0:388])
            self.labels.append(img_obj.ground_truth[176:564, 176:564])

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
