import itertools
import math
import os
from random import shuffle

import keras
import numpy as np

import neuralnet.utils.data_utils as datautils
import utils.img_utils as imgutil
from commons.IMAGE import Image


class KerasPatchesGenerator(keras.utils.Sequence):
    def __init__(self, Dirs=None, batch_size=None, patch_size=None, num_classes=None,
                 fget_mask=None, fget_truth=None, fget_segmented=None, transformation=None):

        """
            :param Dirs: Should contain images, mask, truth, and seegmented path.(segmented only for 4 way classification)
            :param patch_size:
            :param num_classes:
            :param transform:
            :param fget_mask: mask file getter
            :param fget_truth: ground truth file getter
            :param fget_segmented: segmented file getter
        """

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.IDs = []
        self.file_names = os.listdir(Dirs['images'])
        self.images = {}
        self.transformation = transformation
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
                    y = None
                    if self.num_classes == 2:
                        y = 1 if img_obj.ground_truth[i, j] == 255 else 0
                    elif self.num_classes == 4:
                        y = datautils.get_lable(i, j, img_obj.res['segmented'], img_obj.ground_truth)
                    self.IDs.append([ID, i, j, y])

            self.images[ID] = img_obj
        print('### ' + str(self.__len__()) + ' batches found.')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate batches of the data

        IDs_Xes_Yes = self.IDs[index * self.batch_size:(index + 1) * self.batch_size]
        batch = np.empty((self.batch_size, self.patch_size, self.patch_size, 1))
        batch_y = np.empty(self.batch_size)
        # Generate data
        for ix, (ID, i, j, y) in enumerate(IDs_Xes_Yes):
            k_half = int(math.floor(self.patch_size / 2))
            patch = np.full((self.patch_size, self.patch_size), 0, dtype=np.uint8)

            for k in range(-k_half, k_half + 1, 1):
                for l in range(-k_half, k_half + 1, 1):
                    patch_i = i + k
                    patch_j = j + l
                    if self.images[ID].working_arr.shape[0] > patch_i >= 0 and self.images[ID].working_arr.shape[
                        1] > patch_j >= 0:
                        patch[k_half + k, k_half + l] = self.images[ID].working_arr[patch_i, patch_j]

            if self.transformation is not None:
                patch = self.transformation(patch)

            batch[ix, ] = patch[..., None]
            batch_y[ix] = y
        return batch, keras.utils.to_categorical(batch_y, num_classes=self.num_classes)

    def on_epoch_end(self):
        shuffle(self.IDs)
