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
from commons.IMAGE import Image
from neuralnet.datagen import Generator

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.expand_by = self.run_conf.get('Params').get('expand_patch_by')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):
            img_obj = self._get_image_obj(img_file)

            self.indices.append(list(imgutils.get_chunk_indexes(img_obj.working_arr.shape, self.patch_shape,
                                                                self.patch_offset)))
            self.image_objects[ID] = img_obj
        if self.shuffle_indices:
            shuffle(self.indices)

    def _get_image_obj(self, img_file=None):
        img_obj = Image()
        img_obj.load_file(data_dir=self.image_dir,
                          file_name=img_file)
        if self.mask_getter is not None:
            img_obj.load_mask(mask_dir=self.mask_dir,
                              fget_mask=self.mask_getter,
                              erode=True)
        if self.truth_getter is not None:
            img_obj.load_ground_truth(gt_dir=self.truth_dir,
                                      fget_ground_truth=self.truth_getter)

        img_obj.working_arr = imgutils.fix_pad(img_obj.image_arr[:, :, 1], (588, 588))
        img_obj.ground_truth = imgutils.fix_pad(img_obj.ground_truth, (588, 588))
        img_obj.mask = imgutils.fix_pad(img_obj.mask, (588, 588))
        img_obj.apply_clahe()
        img_obj.apply_mask()

        return img_obj

    def __getitem__(self, index):
        chunks = self.indices[index]

        img_obj = self.image_objects[index]

        # Only flip up or down
        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img_obj.working_arr = np.flip(img_obj.working_arr, 0)
            img_obj.ground_truth = np.flip(img_obj.ground_truth, 0)
            img_obj.mask = np.flip(img_obj.mask, 0)

        arr = np.zeros((9, self.patch_shape[0] + self.expand_by[0], self.patch_shape[1] + self.expand_by[1]))
        for i, orig_ix in enumerate(chunks):
            p, q, r, s, pad = imgutils.expand_and_mirror_patch(full_img_shape=img_obj.working_arr.shape,
                                                               orig_patch_indices=orig_ix,
                                                               expand_by=self.expand_by)
            arr[i] = np.pad(img_obj.working_arr[p:q, r:s], pad, 'reflect')

        return {'inputs': arr.copy(),
                'labels': img_obj.ground_truth.copy() / 255}

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
            loader = torch.utils.data.DataLoader(gen, batch_size=min(2, gen.__len__()),
                                                 shuffle=False, num_workers=3, sampler=None)
            loaders.append(loader)
        return loaders
