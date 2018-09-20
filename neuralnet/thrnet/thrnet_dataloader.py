import math
import os
import random
from random import shuffle

import cv2
import numpy as np
from PIL import Image as IMG
from scipy.ndimage.measurements import label
from skimage.morphology import skeletonize

import utils.img_utils as imgutils
from commons.IMAGE import Image
from neuralnet.datagen import Generator
from neuralnet.utils.measurements import get_best_f1_thr

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

            # Load equal number of background patches as well. But only for test set
            if self.mode == 'train':
                all_bg_pix_pos = list(zip(*np.where(img_obj.res['seed_bg'] == 0)))
                shuffle(all_bg_pix_pos)
                all_bg_patch_indices = list(
                    imgutils.get_chunk_indices_by_index(img_obj.res['seed_bg'].shape, self.patch_shape,
                                                        indices=all_bg_pix_pos[0:len(all_patch_indices)]))

                for chunk_ix in all_bg_patch_indices:
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
        sk_mask[::10] = 1
        sk_mask[:, ::10] = 1

        # <PREP5> Apply mask and save seed
        img_obj.res['seed'] = seed * sk_mask * 255

        if self.mode == 'train':
            # <PREP6> NOW WORK ON FINDING equal number of background patch indices
            # No need tp generate background patches for test set
            kernel = np.ones((10, 10), np.uint8)
            dilated_estimate = cv2.dilate(raw_estimate, kernel, iterations=1)
            dilated_estimate[img_obj.mask == 0] = 255
            img_obj.res['seed_bg'] = dilated_estimate

        return img_obj

    def __getitem__(self, index):
        ID, row_from, row_to, col_from, col_to = self.indices[index]

        img_arr = self.image_objects[ID].working_arr.copy()
        gt = self.image_objects[ID].ground_truth.copy()

        prob_map = img_arr[row_from:row_to, col_from:col_to]

        best_score1, best_thr1 = get_best_f1_thr(prob_map, gt[row_from:row_to, col_from:col_to])

        p, q, r, s, pad = imgutils.expand_and_mirror_patch(full_img_shape=img_arr.shape,
                                                           orig_patch_indices=[row_from, row_to, col_from, col_to],
                                                           expand_by=self.expand_by)
        img_tensor = np.pad(img_arr[p:q, r:s], pad, 'reflect')

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img_tensor = np.flip(img_tensor, 0)
            prob_map = np.flip(prob_map, 0)

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img_tensor = np.flip(img_tensor, 1)
            prob_map = np.flip(prob_map, 1)

        img_tensor = img_tensor[..., None]
        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)

        return {'inputs': img_tensor,
                'clip_ix': np.array([row_from, row_to, col_from, col_to]),
                'y_thresholds': best_thr1,
                'prob_map': prob_map.copy()}
