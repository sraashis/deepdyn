import os

import torch
from torch.utils.data.dataset import Dataset

import utils.img_utils as imgutil
from commons.IMAGE import Image


class ImageGenerator(Dataset):
    def __init__(self, Dirs=None, transform=None, fget_mask=None, fget_truth=None, segment_mode=False):
        """
        :param Dirs: Should contain paths to directories images, mask, and truth by the same name.
        :param transform:
        :param fget_mask: mask file getter
        :param fget_truth: ground truth file getter
        :param segment_mode: True only when running a model to segment since it returns the image id and
               pixel positions for each patches. DO NOT USE segment_mode while training or evaluating the model
        """

        self.transform = transform
        self.file_names = os.listdir(Dirs['images'])
        self.segment_mode = segment_mode
        self.fget_mask = fget_mask
        self.fget_truth = fget_truth
        self.Dirs = Dirs
        print('### ' + str(self.__len__()) + ' images found.')

    def __getitem__(self, index):
        img_obj = Image()

        img_obj.load_file(data_dir=self.Dirs['images'], file_name=self.file_names[index])
        img_obj.working_arr = imgutil.whiten_image2d(img_obj.image_arr[:, :, 1])[100:480, 100:480]

        img_obj.load_mask(mask_dir=self.Dirs['mask'], fget_mask=self.fget_mask, erode=True)
        img_obj.load_ground_truth(gt_dir=self.Dirs['truth'], fget_ground_truth=self.fget_truth)
        img_obj.ground_truth = img_obj.ground_truth[100:480, 100:480]
        # img_obj.apply_mask()

        img_tensor = img_obj.working_arr[..., None]
        img_obj.ground_truth[img_obj.ground_truth == 255] = 1

        y_tensor = img_obj.ground_truth[..., None]
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.from_numpy(y_tensor)

    def __len__(self):
        return len(self.file_names)
