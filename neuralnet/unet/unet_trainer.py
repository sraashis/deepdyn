"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os

import numpy as np
import torch
from PIL import Image as IMG

from neuralnet.torchtrainer import NNTrainer
from neuralnet.utils.measurements import ScoreAccumulator

sep = os.sep


class UNetNNTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')

    def evaluate(self, data_loaders=None, logger=None, mode=None, epoch=0):
        assert (logger is not None), 'Please Provide a logger'
        self.model.eval()

        print('\nEvaluating...')
        with torch.no_grad():
            eval_score = ScoreAccumulator()
            for loader in data_loaders:
                img_obj = loader.dataset.image_objects[0]
                segmented_img = torch.cuda.LongTensor(img_obj.working_arr.shape[0],
                                                       img_obj.working_arr.shape[1]).fill_(0)

                for i, data in enumerate(loader, 1):
                    inputs, labels = data['inputs'].float().to(self.device), data['labels'].float().to(self.device)
                    clip_ix = data['clip_ix'].float().to(self.device)

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)

                    for j in range(predicted.shape[0]):
                        p, q, r, s = clip_ix[j]
                        segmented_img[int(p):int(q), int(r):int(s)] += predicted[j]
                    print('Batch: ', i, end='\r')

                segmented_img[segmented_img > 0] = 255
                # segmented_img[img_obj.mask == 0] = 0

                segmented_img = segmented_img.cpu().numpy()
                img_score = ScoreAccumulator()
                img_score.add_array(img_obj.ground_truth, segmented_img)
                eval_score.accumulate(img_score)
                print(img_obj.file_name, ' PRF1A', img_score.get_prf1a())
                IMG.fromarray(np.array(segmented_img, dtype=np.uint8)).save(
                    os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + '.png'))

        if mode is 'train':
            self._save_if_better(score=eval_score.get_prf1a()[2])
