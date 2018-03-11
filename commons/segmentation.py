import os
from itertools import count

import cv2
import numpy as np
from PIL import Image as IMG

import preprocess.algorithms.fast_mst as fmst
import preprocess.utils.filter_utils as fu
import preprocess.utils.img_utils as imgutils
from commons.timer import checktime


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
        img_obj.res['segmented_rgb'] = np.zeros([img_obj.working_arr.shape[0], img_obj.working_arr.shape[1], 3],
                                                dtype=np.uint8)

        img_obj.res['graph'] = self._segment_now(accumulator_2d=img_obj.res['segmented'], image_obj=img_obj,
                                                 params=params)
        img_obj.res['segmented'] = cv2.bitwise_and(img_obj.res['segmented'], img_obj.res['segmented'],
                                                   mask=img_obj.mask)
        img_obj.res['skeleton'] = img_obj.res['skeleton'].copy()
        img_obj.res['params'] = params.copy()
        img_obj.res['scores'] = imgutils.get_praf1(arr_2d=img_obj.res['segmented'], truth=img_obj.ground_truth)

        imgutils.rgb_scores(arr_2d=img_obj.res['segmented'], truth=img_obj.ground_truth,
                            arr_rgb=img_obj.res['segmented_rgb'])

        self._save(img_obj=img_obj, params=params, save_images=save_images)

    def run_all(self, data_dir=None, mask_path=None, gt_path=None, img_obj=None, params_combination=[],
                save_images=False, epochs=1, alpha_decay=0):

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

        def get_mask_file(fn):
            return fn.split('_')[0] + ''

        def get_ground_truth_file(fn):
            return fn.split('_')[0] + ''

        for file_name in os.listdir(data_dir):

            img_obj.load_file(data_dir=data_dir, file_name=file_name)
            img_obj.load_mask(mask_dir=mask_path, fget_mask=get_mask_file, erode=True)
            img_obj.load_ground_truth(gt_dir=gt_path, fget_ground_truth=get_ground_truth_file)

            img_obj.working_arr = img_obj.image_arr[:, :, 1]
            img_obj.apply_mask()

            img_obj.apply_bilateral()
            img_obj.apply_gabor()
            img_obj.generate_lattice_graph(eight_connected=False)

            for params in params_combination:
                self._run(img_obj=img_obj, params=params, save_images=save_images)

        self.writer.close()

    def run(self, params={}, save_images=False, img_obj=None):
        self._run(img_obj=img_obj, params=params, save_images=save_images)

    @checktime
    def _disable_segmented_vessels(self, img_obj=None, params=None, alpha_decay=None):
        # todo something with previous accumulator.img_obj.graph to disable the connectivity
        params['alpha'] -= alpha_decay
        params['sk_threshold'] = 100

    def _save(self, img_obj=None, params=None, epoch=None, save_images=False):
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
                os.path.join(self.out_dir, img_obj.file_name + '_[' + line + ']' + '.JPEG'))
