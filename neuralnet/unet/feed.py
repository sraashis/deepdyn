import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import utils.img_utils as imgutils
from commons.IMAGE import Image


class PatchesGenerator(Dataset):
    def __init__(self, image_files=None, transform=None,
                 fget_mask=None, fget_truth=None, train_image_size=(388, 388), pad_row_col=[(92, 92), (92, 92)]):

        self.transform = transform
        self.patches_indexes = []
        self.images = {}
        self.train_image_size = train_image_size
        self.pad_row_col = pad_row_col
        self.fget_mask = fget_mask
        self.fget_truth = fget_truth
        self.image_files = image_files
        for ID, img_file in enumerate(image_files):
            img_obj = self._get_image_obj(img_file)
            for chunk_ix in imgutils.get_chunk_indexes(img_obj.working_arr.shape, self.train_image_size):
                self.patches_indexes.append([ID] + chunk_ix)
            self.images[ID] = img_obj

        random.shuffle(self.patches_indexes)
        print('### ' + str(self.__len__()) + ' patches found.')

    def _get_image_obj(self, img_file=None):
        img_obj = Image()
        img_obj.load_file(data_dir='images', file_name=img_file)
        img_obj.load_mask(mask_dir='mask', fget_mask=self.fget_mask, erode=True)
        img_obj.load_ground_truth(gt_dir='manual', fget_ground_truth=self.fget_truth)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_obj.working_arr = clahe.apply(img_obj.image_arr[:, :, 1])
        if img_obj.mask is not None:
            x = np.logical_and(True, img_obj.mask == 255)
            img_obj.working_arr[img_obj.mask == 0] = img_obj.working_arr[x].mean()
        return img_obj

    def __getitem__(self, index):
        ID, row_from, row_to, col_from, col_to = self.patches_indexes[index]
        img_tensor = self.images[ID].working_arr[row_from:row_to, col_from:col_to]
        img_tensor = np.pad(img_tensor, self.pad_row_col, 'reflect')
        y = self.images[ID].ground_truth[row_from:row_to, col_from:col_to]

        img_tensor = img_tensor[..., None]
        y[y == 255] = 1
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.LongTensor(y)

    def __len__(self):
        return len(self.patches_indexes)


class UNETDataSet:
    def __init__(self, base_dir=None, transform=None,
                 fget_mask=None, fget_truth=None, train_image_size=(388, 388), pad_row_col=[(92, 92), (92, 92)]):
        self.transform = transform
        self.train_image_size = train_image_size
        self.pad_row_col = pad_row_col
        self.base_dir = base_dir
        self.fget_mask = fget_mask
        self.fget_truth = fget_truth
        self.files = random.shuffle(os.listdir(os.path.join(base_dir, 'images')) if base_dir is not None else [])

    def get_k_fold_loaders(self, num_folds=0, batch_size=4):
        """
        :param num_folds: 0
        :param batch_size: 4
        :param shuffle: False
        :return: train_loader, validation_loader, test_loader of type torch.utils.data.DataLoader
        """
        assert (num_folds > 0), "num_folds must be greater than 0"
        test_set_size = len(self.files) // num_folds
        for k in range(num_folds):
            ix_test_from = test_set_size * k
            ix_test_to = test_set_size * (k + 1)

            test_files = self.files[ix_test_from:ix_test_to]

            train_files = self.files[0:ix_test_from] + self.files[ix_test_to:len(self.files)]
            validation_files = train_files[0:test_set_size]
            train_files = train_files[test_set_size:len(train_files)]

            test_set = PatchesGenerator(
                image_files=test_files,
                transform=self.transform,
                fget_mask=self.fget_mask,
                fget_truth=self.fget_truth,
                train_image_size=self.train_image_size,
                pad_row_col=self.pad_row_col)
            train_set = PatchesGenerator(
                image_files=train_files,
                transform=self.transform,
                fget_mask=self.fget_mask,
                fget_truth=self.fget_truth,
                train_image_size=self.train_image_size,
                pad_row_col=self.pad_row_col
            )
            validation_set = PatchesGenerator(
                image_files=validation_files,
                transform=self.transform,
                fget_mask=self.fget_mask,
                fget_truth=self.fget_truth,
                train_image_size=self.train_image_size,
                pad_row_col=self.pad_row_col
            )

            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=3)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=3)
            validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, num_workers=3)
            yield train_loader, validation_loader, test_loader

    def get_loader(self, file=None):
        data_set = PatchesGenerator(
            image_files=[file],
            transform=self.transform,
            fget_mask=self.fget_mask,
            fget_truth=self.fget_truth,
            train_image_size=self.train_image_size,
            pad_row_col=self.pad_row_col)
        loader = torch.utils.data.DataLoader(data_set, batch_size=4, shuffle=False,
                                             num_workers=3)
        return loader
