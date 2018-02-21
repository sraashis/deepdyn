import os
from itertools import count

import cv2
import numpy as np
from PIL import Image as IMG

import preprocess.algorithms.fast_mst as fmst
import preprocess.utils.img_utils as imgutils
from commons.IMAGE import Image
from commons.MAT import Mat
from commons.accumulator import Accumulator
from commons.timer import checktime


class AtureTest:
    def __init__(self, data_dir=None, out_dir=None):

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.writer = None
        self.mask_dir = None
        self.ground_truth_dir = None
        self.fget_mask_file = None
        self.fget_ground_truth_file = None
        self.erode_mask = None
        self.c = count(1)
        if os.path.isdir(self.out_dir) is False:
            os.makedirs(self.out_dir)

    def load_mask(self, mask_dir=None, fget_mask_file=None, erode_mask=False):
        self.mask_dir = mask_dir
        self.fget_mask_file = fget_mask_file
        self.erode_mask = erode_mask

    def load_ground_truth(self, ground_truth_dir=None, fget_ground_truth_file=None):
        self.ground_truth_dir = ground_truth_dir
        self.fget_ground_truth_file = fget_ground_truth_file

    @checktime
    def _segment_now(self, accumulator_2d=None, image_obj=None, params={}):
        image_obj.create_skeleton(threshold=params['sk_threshold'],
                                  kernels=imgutils.get_chosen_skeleton_filter())
        seed_node_list = imgutils.get_seed_node_list(image_obj.img_skeleton)

        fmst.run_segmentation(accumulator_2d=accumulator_2d, image_obj=image_obj, seed_list=seed_node_list,
                              params=params)

    def _load_file(self, file_name=None):
        img = IMG.open(os.path.join(self.data_dir, file_name))
        orig = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
        print('### File loaded: ' + file_name)
        return orig

    def _load_mask(self, image_obj=None):

        try:
            mask_file = self.fget_mask_file(image_obj.file_name)
            mask = IMG.open(os.path.join(self.mask_dir, mask_file))
            mask = np.array(mask.getdata(), np.uint8).reshape(mask.size[1], mask.size[0], 1)[:, :, 0]

            if self.erode_mask:
                kern = np.array([
                    [0.0, 0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.2, 1.0, 0.2, 0.0],
                    [0.5, 1.0, 1.5, 1.0, 0.5],
                    [0.0, 0.2, 1.0, 0.2, 0.0],
                    [0.0, 0.0, 0.5, 0.0, 0.0],
                ], np.uint8)

                print('Mask loaded: ' + mask_file)
                return cv2.erode(mask, kern, iterations=5)
        except:
            print('!!! Mask not found')
            return np.ones_like(image_obj.img_array)

    def _load_ground_truth(self, image_obj=None):

        try:
            gt_file = self.fget_ground_truth_file(image_obj.file_name)
            truth = IMG.open(os.path.join(self.ground_truth_dir, gt_file))
            truth = np.array(truth.getdata(), np.uint8).reshape(truth.size[1], truth.size[0], 1)[:, :, 0]
            print('Ground truth loaded: ' + gt_file)
            return truth
        except:
            print('!!! Ground truth not found')
            return np.zeros_like(image_obj.img_array)

    @checktime
    def _initialize(self, file_name=None):

        org = self._load_file(file_name=file_name)
        img_obj = Image(image_arr=org[:, :, 1], file_name=file_name)
        mask = self._load_mask(image_obj=img_obj)
        img_obj.img_array = cv2.bitwise_and(img_obj.img_array, img_obj.img_array, mask=mask)

        truth = self._load_ground_truth(image_obj=img_obj)
        img_obj.generate_lattice_graph()
        print('Lattice created')

        return Accumulator(img_obj=img_obj, mask=mask, ground_truth=truth)

    @checktime
    def _generate_rgb(self, arr_2d=None, truth=None, arr_rgb=None):
        for i in range(0, arr_2d.shape[0]):
            for j in range(0, arr_2d.shape[1]):
                if arr_2d[i, j] == 255:
                    arr_rgb[i, j, :] = 255
                if arr_2d[i, j] == 255 and truth[i, j] == 0:
                    arr_rgb[i, j, 0] = 0
                    arr_rgb[i, j, 1] = 255
                    arr_rgb[i, j, 2] = 0
                if arr_2d[i, j] == 0 and truth[i, j] == 255:
                    arr_rgb[i, j, 0] = 255
                    arr_rgb[i, j, 1] = 0
                    arr_rgb[i, j, 2] = 0

    @checktime
    def _calculate_scores(self, arr_2d=None, truth=None):
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(0, arr_2d.shape[0]):
            for j in range(0, arr_2d.shape[1]):
                if arr_2d[i, j] == 255 and truth[i, j] == 255:
                    tp += 1
                if arr_2d[i, j] == 255 and truth[i, j] == 0:
                    fp += 1
                if arr_2d[i, j] == 0 and truth[i, j] == 255:
                    fn += 1
                if arr_2d[i, j] == 0 and truth[i, j] == 0:
                    tn += 1
        p, r, a = 0, 0, 0
        try:
            p = tp / (tp + fp)
        except ZeroDivisionError:
            p = 0

        try:
            r = tp / (tp + fn)
        except ZeroDivisionError:
            r = 0

        try:
            a = (tp + tn) / (tp + fp + fn + tn)
        except ZeroDivisionError:
            a = 0

        return {
            'Precision': p,
            'Recall': r,
            'Accuracy': a,
            'F1': 2 * p * r / (p + r)
        }

    def _run(self, accumulator=None, params={},
             save_images=False, epoch=0):

        current_segmented = np.zeros_like(accumulator.img_obj.img_array)
        current_rgb = np.zeros([accumulator.x_size, accumulator.y_size, 3], dtype=np.uint8)

        self._segment_now(accumulator_2d=current_segmented, image_obj=accumulator.img_obj, params=params)
        current_segmented = cv2.bitwise_and(current_segmented, current_segmented, mask=accumulator.mask)
        accumulator.arr_2d = np.maximum(accumulator.arr_2d, current_segmented)

        accumulator.res['image' + str(epoch)] = accumulator.img_obj.img_array.copy()
        accumulator.res['gabor' + str(epoch)] = accumulator.img_obj.img_gabor.copy()
        accumulator.res['bilateral' + str(epoch)] = accumulator.img_obj.diff_bilateral.copy()
        accumulator.res['skeleton' + str(epoch)] = accumulator.img_obj.img_skeleton.copy()
        accumulator.res['params' + str(epoch)] = params.copy()
        accumulator.res['scores' + str(epoch)] = self._calculate_scores(arr_2d=accumulator.arr_2d,
                                                                        truth=accumulator.ground_truth)
        self._generate_rgb(arr_2d=current_segmented, truth=accumulator.ground_truth, arr_rgb=current_rgb)
        accumulator.res['segmented' + str(epoch)] = current_rgb

        self._generate_rgb(arr_2d=accumulator.arr_2d, truth=accumulator.ground_truth, arr_rgb=accumulator.arr_rgb)
        self._save(accumulator=accumulator, params=params, epoch=epoch, save_images=save_images)

    def run_for_all_images(self, params_combination=[], save_images=False, epochs=1, alpha_decay=0):

        self.writer = open(self.out_dir + os.sep + "segmentation_result.csv", 'w')
        self.writer.write(
            'ITR,EPOCH,FILE_NAME,FSCORE,PRECISION,RECALL,ACCURACY,'
            'SK_THRESHOLD,'
            'ALPHA,'
            'GABOR_CONTRIB,'
            'SEG_THRESHOLD\n'
        )

        for file_name in os.listdir(self.data_dir):
            accumulator = self._initialize(file_name)
            accumulator.img_obj.apply_bilateral()
            accumulator.img_obj.apply_gabor(kernel_bank=imgutils.get_chosen_gabor_bank())
            print('Filter applied')

            for params in params_combination:
                for i in range(epochs):
                    print('Running epoch: ' + str(i))

                    if i > 0:
                        self._disable_segmented_vessels(accumulator=accumulator, params=params, alpha_decay=alpha_decay)
                        accumulator.img_obj.apply_bilateral()
                        accumulator.img_obj.apply_gabor(kernel_bank=imgutils.get_chosen_gabor_bank())
                        print('Filter applied')

                    self._run(accumulator=accumulator, params=params, save_images=save_images, epoch=i)

                # Reset for new parameter combination
                accumulator.arr_2d = np.zeros_like(accumulator.img_obj.img_array)
                accumulator.arr_rgb = np.zeros([accumulator.x_size, accumulator.y_size, 3], dtype=np.uint8)
                accumulator.img_obj.img_array = accumulator.res['image0']

        self.writer.close()

    def run_for_one_image(self, file_name=None, params={}, save_images=False, epochs=1, alpha_decay=0):

        accumulator = self._initialize(file_name)

        for i in range(epochs):
            print('Running epoch: ' + str(i))

            if i > 0:
                self._disable_segmented_vessels(accumulator=accumulator, params=params, alpha_decay=alpha_decay)

            accumulator.img_obj.apply_bilateral()
            accumulator.img_obj.apply_gabor(kernel_bank=imgutils.get_chosen_gabor_bank())
            print('Filter applied')

            self._run(accumulator=accumulator, params=params, save_images=save_images, epoch=i)

        return accumulator

    @checktime
    def _disable_segmented_vessels(self, accumulator=None, params=None, alpha_decay=None):

        for i in range(accumulator.img_obj.img_array.shape[0]):
            for j in range(accumulator.img_obj.img_array.shape[1]):
                if accumulator.arr_2d[i, j] == 255:
                    accumulator.img_obj.img_array[i, j] = 200

        params['alpha'] -= alpha_decay
        params['sk_threshold'] = 100

    @checktime
    def _save(self, accumulator=None, params=None, epoch=None, save_images=False):
        i = next(self.c)
        base = 'scores' + str(epoch)
        line = str(i) + ',' + \
               'EP' + str(epoch) + ',' + \
               str(accumulator.img_obj.file_name) + ',' + \
               str(round(accumulator.res[base]['F1'], 3)) + ',' + \
               str(round(accumulator.res[base]['Precision'], 3)) + ',' + \
               str(round(accumulator.res[base]['Recall'], 3)) + ',' + \
               str(round(accumulator.res[base]['Accuracy'], 3)) + ',' + \
               str(round(params['sk_threshold'], 3)) + ',' + \
               str(round(params['alpha'], 3)) + ',' + \
               str(round(params['gabor_contrib'], 3)) + ',' + \
               str(round(params['seg_threshold'], 3))
        if self.writer is not None:
            self.writer.write(line + '\n')
            self.writer.flush()

        print('Number of params combination tried: ' + str(i))

        if save_images:
            IMG.fromarray(accumulator.arr_rgb).save(
                os.path.join(self.out_dir, accumulator.img_obj.file_name + '_[' + line + ']' + '.JPEG'))
            IMG.fromarray(accumulator.img_obj.img_gabor).save(
                os.path.join(self.out_dir, accumulator.img_obj.file_name + '_[' + line + ']GABOR' + '.JPEG'))
            IMG.fromarray(accumulator.img_obj.img_array).save(
                os.path.join(self.out_dir, accumulator.img_obj.file_name + '_[' + line + ']ORIG' + '.JPEG'))


class AtureTestMat(AtureTest):
    def __init__(self, data_dir=None, out_dir=None):
        super().__init__(data_dir=data_dir, out_dir=out_dir)

    def _load_file(self, file_name=None):
        file = Mat(mat_file=os.path.join(self.data_dir, file_name))
        orig = file.get_image('I2')
        print('File loaded: ' + file_name)
        return orig
