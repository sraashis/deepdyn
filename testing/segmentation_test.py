import os
from itertools import count

import cv2
import numpy as np
from PIL import Image as IMG

import preprocess.algorithms.fast_mst as fmst
import preprocess.utils.img_utils as imgutils
from commons.IMAGE import Image
from commons.ImgLATTICE import Lattice
from commons.MAT import Mat


class AtureTest:
    def __init__(self, data_path=None, out=None):

        self.data_path = data_path
        self.log_path = data_path + os.sep + out

        if os.path.isdir(self.log_path) is False:
            os.makedirs(self.log_path)
            self.log_file = open(self.log_path + os.sep + "segmentation_result.csv", 'w')
        else:
            self.log_file = open(self.log_path + os.sep + "segmentation_result.csv", 'w')

        self.log_file.write(
            'ITERATION,FILE_NAME,FSCORE,PRECISION,RECALL,ACCURACY,' \
            'SKELETONIZE_THRESHOLD,' \
            'IMG_LATTICE_COST_ASSIGNMENT_ALPHA,' \
            'IMG_LATTICE_COST_GABOR_IMAGE_CONTRIBUTION,' \
            'SEGMENTATION_THRESHOLD\n'
        )

        self.mask_path = None
        self.ground_truth_path = None
        self.fget_mask_file = None
        self.fget_ground_truth_file = None
        self.c = count(1)

    def load_mask(self, mask_path=None, fget_mask_file=None):
        self.mask_path = mask_path
        self.fget_mask_file = fget_mask_file

    def load_ground_truth(self, ground_truth_path=None, fget_ground_truth_file=None):
        self.ground_truth_path = ground_truth_path
        self.fget_ground_truth_file = fget_ground_truth_file

    def _segment_now(self, img_obj=None, lattice_obj=None, params={}):
        img_obj.create_skeleton(threshold=params['sk_threshold'], kernels=imgutils.get_chosen_skeleton_filter())
        seed_node_list = imgutils.get_seed_node_list(img_obj.img_skeleton)

        fmst.run_segmentation(image_object=img_obj,
                              lattice_object=lattice_obj,
                              seed_list=seed_node_list,
                              segmentation_threshold=params['seg_threshold'],
                              alpha=params['alpha'],
                              img_gabor_contribution=params['gabor_contrib'],
                              img_original_contribution=1.5 - params['gabor_contrib'])

    def _load_test_file(self, test_file_name=None):
        img = IMG.open(test_file_name)
        orig = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
        print('\n### File loaded: ' + test_file_name)
        return orig

    def _load_mask(self, test_file_name):
        mask = IMG.open(self.fget_mask_file(test_file_name))
        mask = np.array(mask.getdata(), np.uint8).reshape(mask.size[1], mask.size[0], 1)[:, :, 0]
        print('Mask loaded')
        return mask

    def _load_ground_truth(self, test_file_name=None):
        truth = IMG.open(self.fget_ground_truth_file(test_file_name))
        truth = np.array(truth.getdata(), np.uint8).reshape(truth.size[1], truth.size[0], 1)[:, :, 0]
        print('Ground truth loaded')
        return truth

    def _preprocess(self, test_file_name=None):

        os.chdir(self.data_path)
        original = self._load_test_file(test_file_name=test_file_name)
        use_this = original[:, :, 1]
        try:
            os.chdir(self.mask_path)
            mask = self._load_mask(test_file_name=test_file_name)
            use_this = cv2.bitwise_and(use_this, use_this, mask=mask)
        except:
            print('!!! Mask not found')

        img_obj = Image(image_arr=use_this)

        try:
            os.chdir(self.ground_truth_path)
            truth = self._load_ground_truth(test_file_name=test_file_name)
        except:
            truth = None
            print('!!! Ground truth not found')

        img_obj.apply_bilateral()
        img_obj.apply_gabor(kernel_bank=imgutils.get_chosen_gabor_bank())
        print('Filter applied')

        lattice_obj = Lattice(image_arr_2d=img_obj.img_gabor)
        lattice_obj.generate_lattice_graph()
        print('Lattice created')

        return img_obj, lattice_obj, truth

    def precision_recall_accuracy(self, segmented=None, truth=None, segmented_rgb=None):

        if truth is None:
            segmented_rgb[:, :, 0] = segmented
            segmented_rgb[:, :, 1] = segmented
            segmented_rgb[:, :, 2] = segmented
            return 0.01, 0.01, 0.01

        TP = 0  # True Positive
        FP = 0  # False Positive
        FN = 0  # False Negative
        TN = 0  # True Negative
        for i in range(0, segmented.shape[0]):
            for j in range(0, segmented.shape[1]):

                if segmented[i, j] == 255:
                    segmented_rgb[i, j, 0] = 255
                    segmented_rgb[i, j, 1] = 255
                    segmented_rgb[i, j, 2] = 255

                if segmented[i, j] == 255 and truth[i, j] == 255:
                    TP += 1
                if segmented[i, j] == 255 and truth[i, j] == 0:
                    segmented_rgb[i, j, 0] = 0
                    segmented_rgb[i, j, 1] = 255
                    segmented_rgb[i, j, 2] = 0
                    FP += 1
                if segmented[i, j] == 0 and truth[i, j] == 255:
                    segmented_rgb[i, j, 0] = 255
                    segmented_rgb[i, j, 1] = 0
                    segmented_rgb[i, j, 2] = 0
                    FN += 1
                if segmented[i, j] == 0 and truth[i, j] == 0:
                    TN += 1

        return TP / (TP + FP), TP / (TP + FN), (TP + TN) / (TP + FP + FN + TN)

    def _run(self, test_file_name=None, params_combination=[], save_segmentation=False, log=False):

        img_obj, lattice_obj, truth = self._preprocess(test_file_name)

        print('Working...')
        for params in params_combination:

            self._segment_now(img_obj=img_obj, lattice_obj=lattice_obj, params=params)

            segmented_rgb = np.zeros([lattice_obj.accumulator.shape[0], lattice_obj.accumulator.shape[1], 3],
                                     dtype=np.uint8)
            precision, recall, accuracy = self.precision_recall_accuracy(segmented=lattice_obj.accumulator,
                                                                         truth=truth,
                                                                         segmented_rgb=segmented_rgb)
            f1_score = 2 * precision * recall / (precision + recall)

            i = next(self.c)
            line = str(i) + ',' + \
                   str(test_file_name) + ',' + \
                   str(round(f1_score, 3)) + ',' + \
                   str(round(precision, 3)) + ',' + \
                   str(round(recall, 3)) + ',' + \
                   str(round(accuracy, 3)) + ',' + \
                   str(round(params['sk_threshold'], 3)) + ',' + \
                   str(round(params['alpha'], 3)) + ',' + \
                   str(round(params['gabor_contrib'], 3)) + ',' + \
                   str(round(params['seg_threshold'], 3))
            if log:
                self.log_file.write(line + '\n')
                self.log_file.flush()

            if save_segmentation:
                os.chdir(self.log_path)
                IMG.fromarray(segmented_rgb).save(test_file_name + '_[' + line + ']' + '.JPEG')
            print('Number of parameter combinations tried: ' + str(i), end='\r')

    def run_for_all_images(self, params_combination=[], save=False):
        os.chdir(self.data_path)
        for test_file_name in os.listdir(os.getcwd()):
            self._run(test_file_name=test_file_name, params_combination=params_combination, save_segmentation=save,
                      log=True)
        self.log_file.close()

    def run_for_one_image(self, test_file_name=None, params_combination=[], save=False):
        os.chdir(self.data_path)
        self._run(test_file_name=test_file_name, params_combination=params_combination, save_segmentation=save,
                  log=True)
        self.log_file.close()


class AtureTestMat(AtureTest):
    def __init__(self, data_path=None, out=None):
        super().__init__(data_path=data_path, out=out)

    def _load_test_file(self, test_file_name=None):
        file = Mat(file_name=test_file_name)
        orig = file.get_image('I2')
        print('File loaded: ' + test_file_name)
        return orig


class AtureTestErode(AtureTest):
    def __init__(self, data_path=None, out=None):
        super().__init__(data_path=data_path, out=out)

    def _load_mask(self, test_file_name):
        mask = IMG.open(self.fget_mask_file(test_file_name))
        mask = np.array(mask.getdata(), np.uint8).reshape(mask.size[1], mask.size[0], 1)[:, :, 0]
        print('Mask loaded.')

        kern = np.array([
            [0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.2, 1.0, 0.2, 0.0],
            [0.5, 1.0, 1.0, 1.0, 0.5],
            [0.0, 0.2, 1.0, 0.2, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0],
        ], np.uint8)

        return cv2.erode(mask, kern, iterations=5)


class AtureTestMatErode(AtureTest):
    def __init__(self, data_path=None, out=None):
        super().__init__(data_path=data_path, out=out)

    def _load_test_file(self, test_file_name=None):
        file = Mat(file_name=test_file_name)
        orig = file.get_image('I2')
        print('File loaded: ' + test_file_name)
        return orig

    def _load_mask(self, test_file_name):
        mask = IMG.open(self.fget_mask_file(test_file_name))
        mask = np.array(mask.getdata(), np.uint8).reshape(mask.size[1], mask.size[0], 1)[:, :, 0]
        print('Mask loaded.')

        kern = np.array([
            [0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.2, 1.0, 0.2, 0.0],
            [0.5, 1.0, 1.0, 1.0, 0.5],
            [0.0, 0.2, 1.0, 0.2, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0],
        ], np.uint8)

        return cv2.erode(mask, kern, iterations=5)
