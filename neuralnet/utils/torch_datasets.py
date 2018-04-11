import itertools
import math
import os
from random import shuffle

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import neuralnet.utils.data_utils as datautils
import utils.img_utils as imgutil
from commons.IMAGE import Image


class TorchPatchesGenerator(Dataset):
    def __init__(self, Dirs=None, patch_size=None, num_classes=None, transform=None,
                 fget_mask=None, fget_truth=None, fget_segmented=None, segment_mode=False):

        """
        :param Dirs: Should contail images, mask, truth, and seegmented path.(segmented only for 4 way classification)
        :param patch_size:
        :param num_classes:
        :param transform:
        :param fget_mask: mask file getter
        :param fget_truth: ground truth file getter
        :param fget_segmented: segmented file getter
        :param segment_mode: True only when running a model to segment since it returns the image id and
               pixel positions for each patches. DO NOT USE segment_mode while training or evaluating the model
        """

        self.transform = transform
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.data = []
        self.file_names = os.listdir(Dirs['images'])
        self.images = {}
        self.segment_mode = segment_mode
        for ID, img_file in enumerate(self.file_names):

            img_obj = Image()

            img_obj.load_file(data_dir=Dirs['images'], file_name=img_file)
            img_obj.working_arr = imgutil.whiten_image2d(img_obj.image_arr[:, :, 1])

            img_obj.load_mask(mask_dir=Dirs['mask'], fget_mask=fget_mask, erode=True)
            img_obj.load_ground_truth(gt_dir=Dirs['truth'], fget_ground_truth=fget_truth)

            if self.num_classes == 4:
                segmented_file = os.path.join(Dirs['segmented'], fget_segmented(img_file))
                img_obj.res['segmented'] = imgutil.get_image_as_array(segmented_file, channels=1)

            for i, j in itertools.product(np.arange(img_obj.working_arr.shape[0]),
                                          np.arange(img_obj.working_arr.shape[1])):
                if img_obj.mask[i, j] == 255:
                    self.data.append([ID, i, j])

            self.images[ID] = img_obj

        shuffle(self.data)
        self.labels = np.zeros(self.__len__(), dtype=np.uint8)

        for i, IDXY in enumerate(self.data, 0):
            ID, x, y = IDXY
            if self.num_classes == 2:
                if self.images[ID].ground_truth[x, y] == 255:
                    self.labels[i] = 1
                else:
                    self.labels[i] = 0

            elif self.num_classes == 4:
                self.labels[i] = datautils.get_lable(x, y, self.images[ID].res['segmented'],
                                                     self.images[ID].ground_truth)

        self.labels = torch.from_numpy(self.labels)
        print('### ' + str(self.__len__()) + ' data items found.')

    def __getitem__(self, index):
        ID, i, j = self.data[index]
        k_half = int(math.floor(self.patch_size / 2))
        patch = np.full((self.patch_size, self.patch_size), 0, dtype=np.uint8)

        for k in range(-k_half, k_half + 1, 1):
            for l in range(-k_half, k_half + 1, 1):
                patch_i = i + k
                patch_j = j + l
                if self.images[ID].working_arr.shape[0] > patch_i >= 0 and self.images[ID].working_arr.shape[
                    1] > patch_j >= 0:
                    patch[k_half + k, k_half + l] = self.images[ID].working_arr[patch_i, patch_j]

        img_tensor = patch[..., None]
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        if self.segment_mode:
            return self.data[index], img_tensor, self.labels[index]

        return img_tensor, self.labels[index]

    def __len__(self):
        return len(self.data)