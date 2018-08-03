import itertools
import math
import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from neuralnet.datagen import Generator

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, shape=(51, 51), **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.shape = shape
        self.k_half = int(math.floor(self.shape[0] / 2))
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):

            #  SKIP flipped versions
            if img_file[0].isalpha():
                continue

            img_obj = self._get_image_obj(img_file)
            for i, j in itertools.product(np.arange(img_obj.working_arr.shape[0]),
                                          np.arange(img_obj.working_arr.shape[1])):
                row_from, row_to = i - self.k_half, i + self.k_half + 1
                col_from, col_to = j - self.k_half, j + self.k_half + 1
                # Discard all indices that exceeds the image boundary ####
                if row_from < 0 or col_from < 0:
                    continue
                if row_to >= img_obj.working_arr.shape[0] or col_to >= img_obj.working_arr.shape[1]:
                    continue
                # Discard if the pixel (i, j) is not within the mask ###
                if img_obj.mask is not None and img_obj.mask[i, j] != 255:
                    continue
                self.indices.append([ID, i, j, 1 if img_obj.ground_truth[i, j] == 255 else 0])
            self.image_objects[ID] = img_obj

    def __getitem__(self, index):
        ID, i, j, y = self.indices[index]
        row_from, row_to = i - self.k_half, i + self.k_half + 1
        col_from, col_to = j - self.k_half, j + self.k_half + 1

        img_tensor = self.image_objects[ID].working_arr[row_from:row_to, col_from:col_to][..., None]

        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)

        return ID, np.array([i, j]), img_tensor, y

    def get_loader(self, batch_size=512, shuffle=True, sampler=None, num_workers=2):
        return torch.utils.data.DataLoader(self, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=num_workers, sampler=sampler)


def get_loaders(images_dir=None, mask_dir=None, manual_dir=None,
                transform=None, get_mask=None, get_truth=None):
    loaders = []
    for file in os.listdir(images_dir):
        loaders.append(PatchesGenerator(
            images_dir=images_dir,
            image_files=file,
            mask_dir=mask_dir,
            manual_dir=manual_dir,
            transforms=transform,
            get_mask=get_mask,
            get_truth=get_truth
        ).get_loader(shuffle=False))
    return loaders


def split_drive_dataset(Dirs=None, transform=None, batch_size=None):
    for k, folder in Dirs.items():
        os.makedirs(folder, exist_ok=True)

    def get_mask_file(file_name):
        return file_name.split('_')[0] + '_training_mask.gif'

    def get_ground_truth_file(file_name):
        return file_name.split('_')[0] + '_manual1.gif'

    def get_mask_file_test(file_name):
        return file_name.split('_')[0] + '_test_mask.gif'

    train_loader = PatchesGenerator(
        images_dir=Dirs['train'] + sep + 'images',
        mask_dir=Dirs['train'] + sep + 'mask',
        manual_dir=Dirs['train'] + sep + '1st_manual',
        transforms=transform,
        get_mask=get_mask_file,
        get_truth=get_ground_truth_file
    ).get_loader(batch_size=batch_size)

    val_loaders = get_loaders(
        images_dir=Dirs['test'] + sep + 'validation_images',
        mask_dir=Dirs['test'] + sep + 'mask',
        manual_dir=Dirs['test'] + sep + '1st_manual',
        transform=transform,
        get_mask=get_mask_file_test,
        get_truth=get_ground_truth_file
    )

    test_loaders = get_loaders(
        images_dir=Dirs['test'] + sep + 'images',
        mask_dir=Dirs['test'] + sep + 'mask',
        manual_dir=Dirs['test'] + sep + '1st_manual',
        transform=transform,
        get_mask=get_mask_file_test,
        get_truth=get_ground_truth_file
    )

    return train_loader, val_loaders, test_loaders


def split_wide_dataset(Dirs=None, transform=None, batch_size=None):
    for k, folder in Dirs.items():
        os.makedirs(folder, exist_ok=True)

    def get_ground_truth_file(file_name):
        return file_name.split('.')[0] + '_vessels.png'

    train_loader = PatchesGenerator(
        images_dir=Dirs['train'] + sep + 'images',
        mask_dir=Dirs['train'] + sep + 'mask',
        manual_dir=Dirs['train'] + sep + '1st_manual',
        transforms=transform,
        get_truth=get_ground_truth_file
    ).get_loader(batch_size=batch_size)

    val_loaders = get_loaders(
        images_dir=Dirs['test'] + sep + 'validation_images',
        mask_dir=Dirs['test'] + sep + 'mask',
        manual_dir=Dirs['test'] + sep + '1st_manual',
        transform=transform,
        get_truth=get_ground_truth_file
    )

    test_loaders = get_loaders(
        images_dir=Dirs['test'] + sep + 'images',
        mask_dir=Dirs['test'] + sep + 'mask',
        manual_dir=Dirs['test'] + sep + '1st_manual',
        transform=transform,
        get_truth=get_ground_truth_file
    )

    return train_loader, val_loaders, test_loaders
