import os
from random import shuffle

import numpy as np
import torch

import utils.img_utils as imgutils
from neuralnet.datagen import Generator

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, shape=(388, 388), pad=[(92, 92), (92, 92)], **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.shape = shape
        self.pad = pad
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):
            img_obj = self._get_image_obj(img_file)
            for chunk_ix in imgutils.get_chunk_indexes(img_obj.working_arr.shape, self.shape):
                self.indices.append([ID] + chunk_ix)
            self.image_objects[ID] = img_obj
        if self.shuffle:
            shuffle(self.indices)

    def __getitem__(self, index):
        ID, row_from, row_to, col_from, col_to = self.indices[index]
        img_tensor = self.image_objects[ID].working_arr[row_from:row_to, col_from:col_to]
        img_tensor = np.pad(img_tensor, self.pad, 'reflect')
        y = self.image_objects[ID].ground_truth[row_from:row_to, col_from:col_to]
        img_tensor = img_tensor[..., None]
        y[y == 255] = 1
        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)
        return ID, img_tensor, torch.LongTensor(y)

    def get_loader(self, batch_size=8, shuffle=True, sampler=None, num_workers=2):
        return torch.utils.data.DataLoader(self, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=num_workers, sampler=sampler)


def get_loader_per_img(images_dir=None, mask_dir=None, manual_dir=None,
                       transforms=None, get_mask=None, get_truth=None, patch_shape=None):
    loaders = []
    for file in os.listdir(images_dir):
        loaders.append(PatchesGenerator(
            images_dir=images_dir,
            image_files=file,
            mask_dir=mask_dir,
            manual_dir=manual_dir,
            transforms=transforms,
            get_mask=get_mask,
            get_truth=get_truth,
            patch_shape=patch_shape,
            offset_shape=patch_shape,
            shuffle=False
        ).get_loader(shuffle=False))
    return loaders
