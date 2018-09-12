import os
import random
from random import shuffle

import numpy as np

import utils.img_utils as imgutils
from commons.IMAGE import Image
from neuralnet.datagen import Generator
from neuralnet.utils.measurements import get_best_f1_thr

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):

            img_obj = self._get_image_obj(img_file)
            for chunk_ix in imgutils.get_chunk_indexes(img_obj.working_arr.shape, self.patch_shape,
                                                       self.patch_offset):
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
        return img_obj

    def __getitem__(self, index):
        ID, row_from, row_to, col_from, col_to = self.indices[index]
        img_tensor = self.image_objects[ID].working_arr[row_from:row_to, col_from:col_to]
        prob_map = img_tensor.copy()
        y = self.image_objects[ID].ground_truth[row_from:row_to, col_from:col_to]
        best_scores, best_thr = get_best_f1_thr(img_tensor, y)

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img_tensor = np.flip(img_tensor, 0)
            y = np.flip(y, 0)

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img_tensor = np.flip(img_tensor, 1)
            y = np.flip(y, 1)

        img_tensor = img_tensor[..., None]
        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)

        return {'ID': ID, 'inputs': img_tensor, 'labels': y.copy(),
                'y_thresholds': best_thr, 'prob_map': prob_map}
