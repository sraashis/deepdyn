"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os
import random
from random import shuffle

import numpy as np
import torch
import torchvision.transforms as tfm

import utils.img_utils as imgutils
from neuralnet.datagen import Generator

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.expand_by = self.run_conf.get('Params').get('expand_patch_by')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')
        self._load_indices()
        self.gen_class_weights()
        print('Patches:', self.__len__())

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):

            img_obj = self._get_image_obj(img_file)
            img_obj.working_arr = img_obj.working_arr[:, :, 1]  # Just use green channel
            for chunk_ix in imgutils.get_chunk_indexes(img_obj.working_arr.shape, self.patch_shape,
                                                       self.patch_offset):
                self.indices.append([ID] + chunk_ix)
            self.image_objects[ID] = img_obj
        if self.shuffle_indices:
            shuffle(self.indices)

    def __getitem__(self, index):
        ID, row_from, row_to, col_from, col_to = self.indices[index]
        # img_tensor = self.image_objects[ID].working_arr[row_from:row_to, col_from:col_to]
        y = self.image_objects[ID].ground_truth[row_from:row_to, col_from:col_to]

        p, q, r, s, pad = imgutils.expand_and_mirror_patch(full_img_shape=self.image_objects[ID].working_arr.shape,
                                                           orig_patch_indices=[row_from, row_to, col_from, col_to],
                                                           expand_by=self.expand_by)
        img_tensor = np.pad(self.image_objects[ID].working_arr[p:q, r:s], pad, 'reflect')

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img_tensor = np.flip(img_tensor, 0)
            y = np.flip(y, 0)

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img_tensor = np.flip(img_tensor, 1)
            y = np.flip(y, 1)

        img_tensor = img_tensor[..., None]
        y[y == 255] = 1
        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)

        return {'id': ID,
                'inputs': img_tensor,
                'labels': y.copy(),
                'dice_labels': np.array([1 - y, y]),
                'clip_ix': np.array([row_from, row_to, col_from, col_to]), }

    @classmethod
    def get_loader_per_img(cls, images, run_conf, mode=None):
        loaders = []
        for file in images:
            gen = cls(
                run_conf=run_conf,
                images=[file],
                transforms=tfm.Compose([tfm.ToPILImage(), tfm.ToTensor()]),
                shuffle_indices=False,
                mode=mode
            )
            loader = torch.utils.data.DataLoader(gen, batch_size=min(1, gen.__len__()),
                                                 shuffle=False, num_workers=3, sampler=None)
            loaders.append(loader)
        return loaders
