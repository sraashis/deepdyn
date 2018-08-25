import os

import numpy as np
import torch

import utils.img_utils as imgutils
from commons.IMAGE import Image
from neuralnet.datagen import Generator
from neuralnet.utils.measurements import get_best_f1_thr

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, patch_shape=None, **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.patch_shape = patch_shape
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):
            img_obj = self._get_image_obj(img_file)
            for chunk_ix in imgutils.get_chunk_indexes(img_obj.working_arr.shape, self.patch_shape):
                self.indices.append([ID] + chunk_ix)
            self.image_objects[ID] = img_obj

    def _get_image_obj(self, img_file=None):
        img_obj = Image()
        img_obj.load_file(data_dir=self.images_dir,
                          file_name=img_file, num_channels=1)
        if self.get_mask is not None:
            img_obj.load_mask(mask_dir=self.mask_dir,
                              fget_mask=self.get_mask,
                              erode=True)
        if self.get_truth is not None:
            img_obj.load_ground_truth(gt_dir=self.manual_dir,
                                      fget_ground_truth=self.get_truth)

        img_obj.working_arr = img_obj.image_arr
        img_obj.working_arr[img_obj.mask == 0] = 0

        if img_obj.mask is not None:
            x = np.logical_and(True, img_obj.mask == 255)
            img_obj.working_arr[img_obj.mask == 0] = img_obj.working_arr[x].mean()
        return img_obj

    def __getitem__(self, index):
        ID, row_from, row_to, col_from, col_to = self.indices[index]
        img_tensor = self.image_objects[ID].working_arr[row_from:row_to, col_from:col_to]
        y = self.image_objects[ID].ground_truth[row_from:row_to, col_from:col_to]
        best_scores, best_thr = get_best_f1_thr(img_tensor, y)
        # IMG.fromarray(img_tensor).save('THR_IMAGE.png')
        # IMG.fromarray(y).save('y_THR.png')
        img_tensor = img_tensor[..., None]
        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)

        # print(best_scores, best_thr)
        y[y == 255] = 1
        return ID, img_tensor, torch.FloatTensor(y), best_thr

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
            patch_shape=patch_shape
        ).get_loader(shuffle=False))
    return loaders
