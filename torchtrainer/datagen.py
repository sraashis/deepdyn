"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import math
import os

import torch
from torch.utils.data.dataset import Dataset

import utils.data_utils as dutils
from utils.img_utils import Image


class Generator(Dataset):
    def __init__(self, conf=None, images=None,
                 transforms=None, shuffle_indices=False, mode=None, **kwargs):

        self.conf = conf
        self.mask_getter = self.conf.get('Funcs').get('mask_getter')
        self.truth_getter = self.conf.get('Funcs').get('truth_getter')
        self.image_dir = self.conf.get('Dirs').get('image')
        self.mask_dir = self.conf.get('Dirs').get('mask')
        self.truth_dir = self.conf.get('Dirs').get('truth')
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
        img_obj.load_file(data_dir=self.image_dir, file_name=img_file)
        if self.mask_getter is not None:
            img_obj.load_mask(mask_dir=self.mask_dir, fget_mask=self.mask_getter)
        if self.truth_getter is not None:
            img_obj.load_ground_truth(gt_dir=self.truth_dir, fget_ground_truth=self.truth_getter)
            img_obj.ground_truth[img_obj.ground_truth == 1] = 255

        img_obj.working_arr = img_obj.image_arr
        img_obj.apply_clahe()
        img_obj.apply_mask()
        return img_obj

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.indices)

    def gen_class_weights(self):

        if self.mode != 'train':
            return

        self.conf['Params']['cls_weights'] = [0, 0]
        for _, obj in self.image_objects.items():
            cls_weights = dutils.get_class_weights(obj.ground_truth)
            self.conf['Params']['cls_weights'][0] += cls_weights[0]
            self.conf['Params']['cls_weights'][1] += cls_weights[255]

        self.conf['Params']['cls_weights'][0] = self.conf['Params']['cls_weights'][0] / len(self.image_objects)
        self.conf['Params']['cls_weights'][1] = self.conf['Params']['cls_weights'][1] / len(self.image_objects)

    @classmethod
    def get_loader(cls, images, conf, transforms, mode, batch_sizes=[]):
        """
        ###### GET list dataloaders of different batch sizes as specified in batch_sizes
        :param images: List of images for which the torch dataloader will be generated
        :param conf: JSON file. see runs.py
        :param transforms: torchvision composed transforms
        :param mode: 'train' or 'test'
        :param batch_sizes: Default will pick from runs.py. List of integers(batch_size)
                will generate a loader for each batch size
        :return: loader if batch_size is default else list of loaders
        """
        batch_sizes = [conf['Params']['batch_size']] if len(batch_sizes) == 0 else batch_sizes
        gen = cls(conf=conf, images=images, transforms=transforms, shuffle_indices=True, mode=mode)

        dls = []
        for bz in batch_sizes:
            dls.append(torch.utils.data.DataLoader(gen, batch_size=bz, shuffle=True, num_workers=0, sampler=None,
                                                   drop_last=True))
        return dls if len(dls) > 1 else dls[0]

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
            loader = torch.utils.data.DataLoader(gen, batch_size=min(conf['Params']['batch_size'], gen.__len__()),
                                                 shuffle=False, num_workers=3, sampler=None)
            loaders.append(loader)
        return loaders

    @classmethod
    def random_split(cls, images, conf, transforms, mode, size_ratio=[0.8, 0.2]):
        FULL_SIZE = 3
        gen = cls(
            conf=conf,
            images=images,
            transforms=transforms,
            shuffle_indices=False,
            mode=mode
        )
        size_a = math.ceil(size_ratio[0] * len(gen))
        size_b = math.floor(size_ratio[1] * len(gen))

        if len(size_ratio) == FULL_SIZE:
            size_c = len(gen) - (size_a + size_b)

            dataset_a, dataset_b, dataset_c = torch.utils.data.dataset.random_split(gen, [size_a, size_b, size_c])

            loader_a = torch.utils.data.DataLoader(dataset_a,
                                                   batch_size=min(conf['Params']['batch_size'], dataset_a.__len__()),
                                                   shuffle=True, num_workers=3, drop_last=False)
            loader_b = torch.utils.data.DataLoader(dataset_b,
                                                   batch_size=min(conf['Params']['batch_size'], dataset_b.__len__()),
                                                   shuffle=True, num_workers=3, drop_last=False)
            loader_c = torch.utils.data.DataLoader(dataset_c,
                                                   batch_size=min(conf['Params']['batch_size'], dataset_c.__len__()),
                                                   shuffle=True, num_workers=3, drop_last=False)
            return loader_a, loader_b, loader_c

        dataset_a, dataset_b = torch.utils.data.dataset.random_split(gen, [size_a, size_b])

        loader_a = torch.utils.data.DataLoader(dataset_a,
                                               batch_size=min(conf['Params']['batch_size'], dataset_a.__len__(), 2),
                                               shuffle=True, num_workers=3, drop_last=False)
        loader_b = torch.utils.data.DataLoader(dataset_b,
                                               batch_size=min(conf['Params']['batch_size'], dataset_b.__len__(), 2),
                                               shuffle=True, num_workers=3, drop_last=False)
        return loader_a, loader_b
