"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os

import numpy as np
import torch
import torchvision.transforms as tfm
from torch.utils.data.dataset import Dataset

from commons.IMAGE import Image


class Generator(Dataset):
    def __init__(self, run_conf=None, images=None,
                 transforms=None, shuffle_indices=False, mode=None, **kwargs):

        self.run_conf = run_conf
        self.mask_getter = self.run_conf.get('Funcs').get('mask_getter')
        self.truth_getter = self.run_conf.get('Funcs').get('truth_getter')
        self.image_dir = self.run_conf.get('Dirs').get('image')
        self.mask_dir = self.run_conf.get('Dirs').get('mask')
        self.truth_dir = self.run_conf.get('Dirs').get('truth')
        self.shuffle_indices = shuffle_indices
        self.transforms = transforms
        self.mode = mode

        if images is not None:
            self.images = images
        else:
            self.images = os.listdir(self.image_dir)
        self.image_objects = {}
        self.indices = []

    def _load_indices(self):
        pass

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

        if len(img_obj.image_arr.shape) == 3:
            img_obj.working_arr = img_obj.image_arr[:, :, 1]
        elif len(img_obj.image_arr.shape) == 2:
            img_obj.working_arr = img_obj.image_arr

        img_obj.apply_clahe()
        if img_obj.mask is not None:
            x = np.logical_and(True, img_obj.mask == 255)
            img_obj.working_arr[img_obj.mask == 0] = img_obj.working_arr[x].mean()
        return img_obj

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.indices)

    @classmethod
    def get_loader(cls, images, run_conf, transforms, mode, batch_sizes=[]):
        """
        ###### GET list dataloaders of different batch sizes as specified in batch_sizes
        :param images: List of images for which the torch dataloader will be generated
        :param run_conf: JSON file. see runs.py
        :param transforms: torchvision composed transforms
        :param mode: 'train' or 'test'
        :param batch_sizes: Default will pick from runs.py. List of integers(batch_size)
                will generate a loader for each batch size
        :return: loader if batch_size is default else list of loaders
        """
        batch_sizes = [run_conf['Params']['batch_size']] if len(batch_sizes) == 0 else batch_sizes
        gen = cls(run_conf=run_conf, images=images, transforms=transforms, shuffle_indices=True, mode=mode)

        dls = []
        for bz in batch_sizes:
            dls.append(torch.utils.data.DataLoader(gen, batch_size=bz, shuffle=True, num_workers=5, sampler=None))
        return dls if len(dls) > 1 else dls[0]

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
            loader = torch.utils.data.DataLoader(gen, batch_size=min(run_conf['Params']['batch_size'], gen.__len__()),
                                                 shuffle=False, num_workers=3, sampler=None)
            loaders.append(loader)
        return loaders
