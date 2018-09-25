import itertools
import math
import os
from random import shuffle

import numpy as np

from neuralnet.datagen import Generator

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.k_half = int(math.floor(self.patch_shape[0] / 2))
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):

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
        if self.shuffle:
            shuffle(self.indices)

    def __getitem__(self, index):
        ID, i, j, y = self.indices[index]
        row_from, row_to = i - self.k_half, i + self.k_half + 1
        col_from, col_to = j - self.k_half, j + self.k_half + 1

        img_tensor = self.image_objects[ID].working_arr[row_from:row_to, col_from:col_to][..., None]

        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)

        return {'IDs': ID, 'IJs': np.array([i, j]), 'inputs': img_tensor, 'labels': y}
