"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os
from random import shuffle

import numpy as np
import torch
import torchvision.transforms as tfm
from skimage.morphology import skeletonize

import utils.img_utils as iu
from utils.img_utils import Image
from nbee.datagen import Generator
import random

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.patch_shape = self.conf.get('Params').get('patch_shape')
        self.expand_by = self.conf.get('Params').get('expand_patch_by')
        self.patch_offset = self.conf.get('Params').get('patch_offset')
        self.unet_dir = self.conf['Dirs']['image_unet']
        self.input_image_ext = '.png'
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):

            img_obj = self._get_image_obj(img_file)
            all_pix_pos = list(zip(*np.where(img_obj.extra['seed'] == 255)))
            all_patch_indices = list(
                iu.get_chunk_indices_by_index(img_obj.working_arr.shape, self.patch_shape, all_pix_pos))
            # all_patch_indices = list(
            #     iu.get_chunk_indexes(img_obj.working_arr.shape, self.patch_shape, self.patch_shape))
            for chunk_ix in all_patch_indices:
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

        img_obj.working_arr = img_obj.image_arr[:, :, 1]
        img_obj.apply_clahe()
        img_obj.apply_mask()

        sup, res = 20, 235

        img_obj.extra['unet'] = iu.get_image_as_array(
            self.unet_dir + sep + img_obj.file_name.split('.')[0] + self.input_image_ext, 1)

        img_obj.extra['indices'] = list(zip(*np.where((img_obj.extra['unet'] >= sup) & (img_obj.extra['unet'] <= res))))

        img_obj.extra['fill_in'] = np.zeros_like(img_obj.working_arr)
        img_obj.extra['fill_in'][img_obj.extra['unet'] > res] = 1

        img_obj.extra['mid_pix'] = img_obj.extra['unet'].copy()
        img_obj.extra['mid_pix'][img_obj.extra['mid_pix'] < sup] = 0
        img_obj.extra['mid_pix'][img_obj.extra['mid_pix'] > res] = 0

        img_obj.extra['gt_mid'] = img_obj.ground_truth.copy()
        img_obj.extra['gt_mid'][img_obj.extra['unet'] > res] = 0
        img_obj.extra['gt_mid'][img_obj.extra['unet'] < sup] = 0

        # <PREP1> Segment with a low threshold and get a raw segmented image
        raw_estimate = img_obj.extra['unet'].copy()
        raw_estimate[raw_estimate > sup] = 255
        raw_estimate[raw_estimate <= sup] = 0

        # <PREP2> Clear up small components(components less that 20px)
        raw_estimate = iu.remove_connected_comp(raw_estimate.squeeze(), 10)

        # <PREP3> Skeletonize binary image
        seed = raw_estimate.copy()
        seed[seed == 255] = 1
        seed = skeletonize(seed).astype(np.uint8)

        # <PREP4> Come up with a grid mask to select few possible pixels to reconstruct the vessels from
        sk_mask = np.zeros_like(seed)
        sk_mask[::int(0.6 * self.patch_shape[0])] = 1
        sk_mask[:, ::int(0.6 * self.patch_shape[0])] = 1

        # <PREP5> Apply mask and save seed
        img_obj.extra['seed'] = seed * sk_mask * 255

        return img_obj

    def __getitem__(self, index):
        ID, row_from, row_to, col_from, col_to = self.indices[index]

        orig = self.image_objects[ID].working_arr
        unet_map = 255 - self.image_objects[ID].extra['unet']
        mid_pix = 255 - self.image_objects[ID].extra['mid_pix']

        y_mid = self.image_objects[ID].extra['gt_mid'][row_from:row_to, col_from:col_to]
        p, q, r, s, pad = iu.expand_and_mirror_patch(full_img_shape=self.image_objects[ID].working_arr.shape,
                                                     orig_patch_indices=[row_from, row_to, col_from, col_to],
                                                     expand_by=self.expand_by)
        orig_patch = np.pad(orig[p:q, r:s], pad, 'reflect')
        mid_patch = np.pad(mid_pix[p:q, r:s], pad, 'reflect')
        unet_patch = np.pad(unet_map[p:q, r:s], pad, 'reflect')

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            orig_patch = np.flip(orig_patch, 0)
            unet_patch = np.flip(unet_patch, 0)
            mid_patch = np.flip(mid_patch, 0)
            y_mid = np.flip(y_mid, 0)

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            orig_patch = np.flip(orig_patch, 1)
            unet_patch = np.flip(unet_patch, 1)
            mid_patch = np.flip(mid_patch, 1)
            y_mid = np.flip(y_mid, 1)

        y_mid[y_mid == 255] = 1
        if self.conf['Params']['num_channels'] == 1:
            img_tensor = np.array([mid_patch])
        else:
            img_tensor = np.array([mid_patch, unet_patch])

        return {'id': ID,
                'inputs': img_tensor,
                'labels': y_mid.copy(),
                'clip_ix': np.array([row_from, row_to, col_from, col_to]), }

    @classmethod
    def get_loader_per_img(cls, images, conf, mode, transforms):
        loaders = []
        for file in images:
            gen = cls(
                conf=conf,
                images=[file],
                transforms=transforms,
                shuffle_indices=False,
                mode=mode
            )
            loader = torch.utils.data.DataLoader(gen, batch_size=min(16, gen.__len__()),
                                                 shuffle=False, num_workers=3, sampler=None)
            loaders.append(loader)
        return loaders
