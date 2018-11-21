import math
import os
from random import shuffle

import numpy as np

import utils.img_utils as iu
from commons.IMAGE import Image
from neuralnet.datagen import Generator
import torch
import torchvision.transforms as tfm

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.k_half = int(math.floor(self.patch_shape[0] / 2))
        self.unet_dir = self.run_conf['Dirs']['image_unet']
        self._load_indices()
        print('Patches:', self.__len__())

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

        img_obj.working_arr = img_obj.image_arr[:, :, 1]
        img_obj.apply_clahe()
        img_obj.apply_mask()

        sup, res = 20, 235

        img_obj.res['unet'] = iu.get_image_as_array(self.unet_dir + sep + img_obj.file_name.split('.')[0] + '.png', 1)

        img_obj.res['indices'] = list(zip(*np.where((img_obj.res['unet'] >= sup) & (img_obj.res['unet'] <= res))))

        img_obj.res['fill_in'] = np.zeros_like(img_obj.working_arr)
        img_obj.res['fill_in'][img_obj.res['unet'] > res] = 1

        img_obj.res['mid_pix'] = img_obj.res['unet'].copy()
        img_obj.res['mid_pix'][img_obj.res['mid_pix'] < sup] = 0
        img_obj.res['mid_pix'][img_obj.res['mid_pix'] > res] = 0

        img_obj.res['gt_mid'] = img_obj.ground_truth.copy()
        img_obj.res['gt_mid'][img_obj.res['unet'] > res] = 0
        img_obj.res['gt_mid'][img_obj.res['unet'] < sup] = 0
        # import PIL.Image as I
        # I.fromarray(img_obj.res['unet']).save('unet.png')
        # I.fromarray(img_obj.res['fill_in']*255).save('fill_in.png')
        # I.fromarray(img_obj.res['mid_pix']).save('mid_pix.png')
        # I.fromarray(img_obj.res['gt_mid']).save('gt.png')
        # i= input('GIVE:')

        return img_obj

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):

            img_obj = self._get_image_obj(img_file)
            for i, j in img_obj.res['indices']:
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
        if self.shuffle_indices:
            shuffle(self.indices)

    def __getitem__(self, index):
        ID, i, j, y = self.indices[index]
        row_from, row_to = i - self.k_half, i + self.k_half + 1
        col_from, col_to = j - self.k_half, j + self.k_half + 1

        orig = self.image_objects[ID].working_arr[row_from:row_to, col_from:col_to]
        unet_map = 255 - self.image_objects[ID].res['unet'][row_from:row_to, col_from:col_to]
        mid_pix = 255 - self.image_objects[ID].res['mid_pix'][row_from:row_to, col_from:col_to]

        return {'IDs': ID, 'IJs': np.array([i, j]),
                'inputs': np.array([mid_pix, unet_map, orig]),
                'labels': y}

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
            loader = torch.utils.data.DataLoader(gen, batch_size=min(256, gen.__len__()),
                                                 shuffle=False, num_workers=3, sampler=None)
            loaders.append(loader)
        return loaders
