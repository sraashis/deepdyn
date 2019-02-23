"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os
from itertools import count

import cv2
import numpy as np
from PIL import Image as IMG

import testing.fast_mst as fmst
import commons.filter_utils as fu
import imgcommons.helper as imgutils
from commons.IMAGE import SegmentedImage, MatSegmentedImage
from commons.timer import checktime

sep = os.sep


class AtureTest:
    def __init__(self, out_dir=None):

        self.out_dir = out_dir
        self.writer = None
        self.c = count(1)

    def _segment_now(self, accumulator_2d=None, image_obj=None, params={}):

        seed_node_list = fu.get_seed_node_list(image_obj.res['skeleton'])

        img_used = [(params['orig_contrib'], image_obj.res['orig']),
                    (1 - params['orig_contrib'], image_obj.working_arr)]

        segmented_graph = image_obj.res['lattice'].copy()
        fmst.run_segmentation(accumulator_2d=accumulator_2d, images_used=img_used, seed_list=seed_node_list,
                              params=params, graph=segmented_graph)

        return segmented_graph

    def _run(self, img_obj=None, params={},
             save_images=False):

        img_obj.res['segmented'] = np.zeros_like(img_obj.working_arr)

        img_obj.res['graph'] = self._segment_now(accumulator_2d=img_obj.res['segmented'], image_obj=img_obj,
                                                 params=params)
        img_obj.res['segmented'] = cv2.bitwise_and(img_obj.res['segmented'], img_obj.res['segmented'],
                                                   mask=img_obj.mask)
        img_obj.res['skeleton'] = img_obj.res['skeleton'].copy()
        img_obj.res['params'] = params.copy()
        img_obj.res['scores'] = imgutils.get_praf1(arr_2d=img_obj.res['segmented'], truth=img_obj.ground_truth)

        self._save(img_obj=img_obj, params=params, save_images=save_images)

    def run_all(self, Dirs=None, fget_mask=None, fget_gt=None, params_combination=[],
                save_images=False):

        if os.path.isdir(self.out_dir) is False:
            os.makedirs(self.out_dir)

        self.writer = open(self.out_dir + os.sep + "segmentation_result.csv", 'w')
        self.writer.write(
            'ITR,FILE_NAME,F1,PRECISION,RECALL,ACCURACY,'
            'SK_THRESHOLD,'
            'ALPHA,'
            'ORIG_CONTRIB,'
            'SEG_THRESHOLD\n'
        )
        for file_name in os.listdir(Dirs['images']):
            print('File: ' + file_name)

            img_obj = SegmentedImage()

            img_obj.load_file(data_dir=Dirs['images'], file_name=file_name)
            img_obj.working_arr = img_obj.image_arr[:, :, 1]
            img_obj.apply_clahe()
            img_obj.res['orig'] = img_obj.working_arr

            img_obj.working_arr = imgutils.get_image_as_array(Dirs['segmented'] + sep + file_name + '.png', channels=1)
            img_obj.load_mask(mask_dir=Dirs['mask'], fget_mask=fget_mask)
            img_obj.load_ground_truth(gt_dir=Dirs['truth'], fget_ground_truth=fget_gt)
            img_obj.apply_mask()
            img_obj.generate_lattice_graph()

            arr = img_obj.working_arr.copy()
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if img_obj.mask[i, j] == 0:
                        arr[i, j] = 255
            img_obj.working_arr = arr

            for params in params_combination:
                img_obj.generate_skeleton(threshold=int(params['sk_threshold']))
                self._run(img_obj=img_obj, params=params, save_images=save_images)

        self.writer.close()

    def run(self, params={}, save_images=False, img_obj=None):
        self._run(img_obj=img_obj, params=params, save_images=save_images)

    @checktime
    def _disable_segmented_vessels(self, img_obj=None, params=None, alpha_decay=None):
        # todo something with previous accumulator.img_obj.graph to disable the connectivity
        params['alpha'] -= alpha_decay
        params['sk_threshold'] = 100

    def _save(self, img_obj=None, params=None, save_images=False):
        i = next(self.c)
        base = 'scores'
        line = str(i) + ',' + \
               str(img_obj.file_name) + ',' + \
               str(round(img_obj.res[base]['F1'], 3)) + ',' + \
               str(round(img_obj.res[base]['Precision'], 3)) + ',' + \
               str(round(img_obj.res[base]['Recall'], 3)) + ',' + \
               str(round(img_obj.res[base]['Accuracy'], 3)) + ',' + \
               str(round(params['sk_threshold'], 3)) + ',' + \
               str(round(params['alpha'], 3)) + ',' + \
               str(round(params['orig_contrib'], 3)) + ',' + \
               str(round(params['seg_threshold'], 3))
        if self.writer is not None:
            self.writer.write(line + '\n')
            self.writer.flush()

        print('Number of params combination tried: ' + str(i))

        if save_images:
            IMG.fromarray(img_obj.res['segmented']).save(
                os.path.join(self.out_dir, img_obj.file_name + '_SEG.PNG'))
            IMG.fromarray(imgutils.get_rgb_scores(img_obj.res['segmented'], img_obj.ground_truth)).save(
                os.path.join(self.out_dir, img_obj.file_name + '_RGB.PNG'))


class AtureTestMat(AtureTest):
    def run_all(self, data_dir=None, mask_path=None, gt_path=None, fget_mask=None, fget_gt=None, params_combination=[],
                save_images=False):

        if os.path.isdir(self.out_dir) is False:
            os.makedirs(self.out_dir)

        self.writer = open(self.out_dir + os.sep + "segmentation_result.csv", 'w')
        self.writer.write(
            'ITR,FILE_NAME,FSCORE,PRECISION,RECALL,ACCURACY,'
            'SK_THRESHOLD,'
            'ALPHA,'
            'ORIG_CONTRIB,'
            'SEG_THRESHOLD\n'
        )

        for file_name in os.listdir(data_dir):
            print('File: ' + file_name)

            img_obj = MatSegmentedImage()

            img_obj.load_file(data_dir=data_dir, file_name=file_name)
            img_obj.res['orig'] = img_obj.image_arr[:, :, 1]
            img_obj.working_arr = img_obj.image_arr[:, :, 1]

            img_obj.load_mask(mask_dir=mask_path, fget_mask=fget_mask)
            img_obj.load_ground_truth(gt_dir=gt_path, fget_ground_truth=fget_gt)

            img_obj.apply_mask()
            img_obj.apply_bilateral()
            img_obj.apply_gabor()

            img_obj.generate_lattice_graph()

            for params in params_combination:
                img_obj.generate_skeleton(threshold=int(params['sk_threshold']))
                self._run(img_obj=img_obj, params=params, save_images=save_images)

        self.writer.close()
