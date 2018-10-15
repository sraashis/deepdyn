import math
import os
from random import shuffle

import PIL.Image as IMG
import cv2
import numpy as np
import torch
from scipy.ndimage.measurements import label
from skimage.morphology import skeletonize

import utils.img_utils as imgutils
from commons.IMAGE import Image
from neuralnet.datagen import Generator
from neuralnet.utils.measurements import get_best_thr
import torchvision.transforms as tfm

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.expand_by = self.run_conf.get('Params').get('expand_patch_by')
        self.est_thr = self.run_conf.get('Params').get('est_threshold', 20)
        self.component_diameter_limit = self.run_conf.get('Params').get('comp_diam_limit', 20)
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):

            img_obj = self._get_image_obj(img_file)

            # Load the patch corners based on estimated pixel seed
            all_pix_pos = list(zip(*np.where(img_obj.res['seed'] == 255)))
            all_patch_indices = list(
                imgutils.get_chunk_indices_by_index(img_obj.res['seed'].shape, self.patch_shape,
                                                    indices=all_pix_pos))
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

        # <PREP1> Segment with a low threshold and get a raw segmented image
        img_obj.working_arr[img_obj.mask == 0] = 0
        raw_estimate = img_obj.working_arr.copy()
        raw_estimate[raw_estimate > self.est_thr] = 255
        raw_estimate[raw_estimate <= self.est_thr] = 0

        # <PREP2> Clear up small components(components less that 20px)
        structure = np.ones((3, 3), dtype=np.int)
        labeled, ncomponents = label(raw_estimate, structure)
        for i in range(ncomponents):
            ixy = np.array(list(zip(*np.where(labeled == i))))
            x1, y1 = ixy[0]
            x2, y2 = ixy[-1]
            dst = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dst < self.component_diameter_limit:
                for u, v in ixy:
                    raw_estimate[u, v] = 0

        # <PREP3> Binarize the image and extract skeleton
        seed = raw_estimate.copy()
        seed[seed == 255] = 1
        seed = skeletonize(seed).astype(np.uint8)

        # <PREP4> Come up with a grid mask to select few possible pixels to reconstruct the vessels from
        sk_mask = np.zeros_like(seed)
        sk_mask[::15] = 1
        sk_mask[:, ::15] = 1

        # <PREP5> Apply mask and save seed
        img_obj.res['seed'] = seed * sk_mask * 255
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
        gt = self.image_objects[ID].ground_truth.copy()

        img_tensor = img_arr[:, row_from:row_to, col_from:col_to]
        y = gt[row_from:row_to, col_from:col_to]
        prob_map = self.image_objects[ID].working_arr[row_from:row_to, col_from:col_to].copy()

        best_score1, best_thr1 = get_best_thr(prob_map, y, for_best='F1')

        y[y == 255] = 1
        return {'inputs': torch.FloatTensor(img_tensor),
                'clip_ix': np.array([row_from, row_to, col_from, col_to]),
                'y_thresholds': best_thr1,
                'prob_map': prob_map,
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