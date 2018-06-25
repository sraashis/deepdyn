import os

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from commons.IMAGE import Image


class Generator(Dataset):
    def __init__(self, images_dir=None, image_files=None, mask_dir=None, manual_dir=None, transforms=None, get_mask=None,
                 get_truth=None, **kwargs):
        """
        :param images_dir: Directory where images, mask and manual1 folders are
        :param image_files: Pass a list of file names inside images.
        :param transforms:
        :param get_mask:
        :param get_truth:
        """
        self.transforms = transforms
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.manual_dir = manual_dir

        self.get_mask = get_mask
        self.get_truth = get_truth

        if image_files is not None:
            self.images = image_files if isinstance(image_files, list) else [image_files]
        else:
            self.images = os.listdir(self.images_dir)
        self.image_objects = {}
        self.indices = []

    def _load_indices(self):
        pass

    def _get_image_obj(self, img_file=None):
        img_obj = Image()
        img_obj.load_file(data_dir=self.images_dir,
                          file_name=img_file)

        img_obj.load_mask(mask_dir=self.mask_dir,
                          fget_mask=self.get_mask,
                          erode=True)

        img_obj.load_ground_truth(gt_dir=self.manual_dir,
                                  fget_ground_truth=self.get_truth)
        img_obj.working_arr = img_obj.image_arr[:, :, 1]
        img_obj.apply_clahe()
        if img_obj.mask is not None:
            x = np.logical_and(True, img_obj.mask == 255)
            img_obj.working_arr[img_obj.mask == 0] = img_obj.working_arr[x].mean()
        return img_obj

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.indices)

    def get_loader(self, batch_size=32, shuffle=True, sampler=None, num_workers=2):
        return torch.utils.data.DataLoader(self, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=num_workers, sampler=sampler)
