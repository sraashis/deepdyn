import os
from random import shuffle

import cv2
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
        self.est_thr = self.run_conf.get('Params').get('est_threshold', 20)
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')
        self.component_diameter_limit = self.run_conf.get('Params').get('comp_diam_limit', 20)
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):

            img_obj = self._get_image_obj(img_file)

            all_patch_indices = list(
                imgutils.get_chunk_indexes(img_obj.working_arr.shape, self.patch_shape, self.patch_offset))
            for chunk_ix in all_patch_indices:
                self.indices.append([ID] + chunk_ix)

            self.image_objects[ID] = img_obj
        if self.shuffle_indices:
            shuffle(self.indices)

    def _get_image_obj(self, img_file=None):
        img_obj = Image()
        img_obj.load_file(data_dir=self.image_dir,
                          file_name=img_file, num_channels=1)
        if self.mask_getter is not None:
            img_obj.load_mask(mask_dir=self.mask_dir,
                              fget_mask=self.mask_getter,
                              erode=True)
        if self.truth_getter is not None:
            img_obj.load_ground_truth(gt_dir=self.truth_dir,
                                      fget_ground_truth=self.truth_getter)

        if len(img_obj.image_arr.shape) == 3:
            img_obj.working_arr = img_obj.image_arr[:, :, 1]
        elif len(img_obj.image_arr.shape) == 2:
            img_obj.working_arr = img_obj.image_arr

        if img_obj.mask is not None:
            x = np.logical_and(True, img_obj.mask == 255)
            img_obj.working_arr[img_obj.mask == 0] = img_obj.working_arr[x].mean()

        img_obj.apply_clahe()
        ks = []
        for i in range(3):
            for j in range(3):
                k = np.zeros((3, 3))
                if True:
                    k[i, j] = 1
                    ks.append(k)

        img_obj.res['t9'] = np.zeros((len(ks), *img_obj.working_arr.shape))
        for i in range(len(ks)):
            img_obj.res['t9'][i] = cv2.filter2D(img_obj.working_arr, -1, ks[i])

        return img_obj

    def __getitem__(self, index):
        ID, row_from, row_to, col_from, col_to = self.indices[index]

        img_arr = self.image_objects[ID].res['t9'].copy()
        img_tensor = []
        p, q, r, s, pad = imgutils.expand_and_mirror_patch(full_img_shape=self.image_objects[ID].working_arr.shape,
                                                           orig_patch_indices=[row_from, row_to, col_from, col_to],
                                                           expand_by=self.expand_by)

        for i in range(self.image_objects[ID].res['t9'].shape[0]):
            img_tensor.append(np.pad(img_arr[i][p:q, r:s], pad, 'reflect'))

        img_tensor = np.array(img_tensor)

        y = self.image_objects[ID].ground_truth.copy()[row_from:row_to, col_from:col_to]
        # y[y == 255] = 1
        return {'inputs': torch.FloatTensor(img_tensor),
                'clip_ix': np.array([row_from, row_to, col_from, col_to]),
                'labels': y.copy()}

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
            loader = torch.utils.data.DataLoader(gen, batch_size=min(16, gen.__len__()),
                                                 shuffle=False, num_workers=3, sampler=None)
            loaders.append(loader)
        return loaders
