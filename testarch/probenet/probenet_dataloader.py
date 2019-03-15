"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os
from random import shuffle

import utils.img_utils as imgutils
import numpy as np
import torch
import torchvision.transforms as tfm
from utils.img_utils import Image

from nbee.datagen import Generator

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.patch_shape = self.conf.get('Params').get('patch_shape')
        self.expand_by = self.conf.get('Params').get('expand_patch_by')
        self.patch_offset = self.conf.get('Params').get('patch_offset')
        self.probe_mode = self.conf.get('Params').get('probe_mode')
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):

            img_obj = self._get_image_obj(img_file)

            img_shape = img_obj.working_arr.shape[0], img_obj.working_arr.shape[1]

            for chunk_ix in imgutils.get_chunk_indexes(img_shape, self.patch_shape,
                                                       self.patch_offset):
                self.indices.append([ID] + chunk_ix)
            self.image_objects[ID] = img_obj
        if self.shuffle_indices:
            shuffle(self.indices)

    def _get_image_obj(self, img_file=None):
        img_obj = Image()
        img_obj.load_file(data_dir=self.image_dir, file_name=img_file)
        if self.mask_getter is not None:
            img_obj.load_mask(mask_dir=self.mask_dir, fget_mask=self.mask_getter)
        if self.truth_getter is not None:
            img_obj.load_ground_truth(gt_dir=self.truth_dir, fget_ground_truth=self.truth_getter)

        # Input images has four channels
        if self.probe_mode == 'depth':
            img_obj.working_arr = img_obj.image_arr[:, :, 0]
            img_obj.ground_truth = img_obj.ground_truth[:, :, 0]

        elif self.probe_mode == 'normal':
            img_obj.working_arr = img_obj.image_arr[:, :, 0:3]
            img_obj.ground_truth = img_obj.ground_truth[:, :, 0:3]

        img_obj.apply_clahe()
        img_obj.apply_mask()
        return img_obj

    def __getitem__(self, index):
        ID, row_from, row_to, col_from, col_to = self.indices[index]

        img_obj = self.image_objects[ID]
        img_shape = img_obj.working_arr.shape
        p, q, r, s, pad = imgutils.expand_and_mirror_patch(full_img_shape=(img_shape[0], img_shape[1]),
                                                           orig_patch_indices=[row_from, row_to, col_from, col_to],
                                                           expand_by=self.expand_by)

        if self.probe_mode == 'depth':
            img_tensor = np.pad(img_obj.working_arr[p:q, r:s], pad, 'reflect')
            y = img_obj.ground_truth[row_from:row_to, col_from:col_to]

        elif self.probe_mode == 'normal':
            img_tensor = np.zeros((3, 572, 572))
            y = np.zeros((3, self.patch_shape[0], self.patch_shape[1]))
            for i in range(img_shape[2]):
                img_tensor[i, :, :] = np.pad(img_obj.working_arr[:, :, i][p:q, r:s], pad, 'reflect')
                y[i, :, :] = img_obj.ground_truth[row_from:row_to, col_from:col_to, i]

        # if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
        #     img_tensor = np.flip(img_tensor, 0)
        #     y = np.flip(y, 0)
        #
        # if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
        #     img_tensor = np.flip(img_tensor, 1)
        #     y = np.flip(y, 1)

        # if self.transforms is not None:
        #     img_tensor = self.transforms(img_tensor)

        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor[None, ...]
        return {'id': ID,
                'inputs': img_tensor,
                'labels': y.copy(),
                'clip_ix': np.array([row_from, row_to, col_from, col_to]), }

    @classmethod
    def get_loader_per_img(cls, images, conf, mode=None):
        loaders = []
        for file in images:
            gen = cls(
                conf=conf,
                images=[file],
                transforms=tfm.Compose([tfm.ToPILImage(), tfm.ToTensor()]),
                shuffle_indices=False,
                mode=mode
            )
            loader = torch.utils.data.DataLoader(gen, batch_size=min(1, gen.__len__()),
                                                 shuffle=False, num_workers=3, sampler=None)
            loaders.append(loader)
        return loaders
