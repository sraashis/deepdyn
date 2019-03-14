"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os

import numpy as np
import torch
from PIL import Image as IMG

from nnbee.torchbee import NNBee
from nnbee.utils.measurements import ScoreAccumulator

sep = os.sep


class UNetBee(NNBee):
    def __init__(self, **kwargs):
        NNBee.__init__(self, **kwargs)
        self.patch_shape = self.conf.get('Params').get('patch_shape')
        self.patch_offset = self.conf.get('Params').get('patch_offset')

    def log_header(self):
        return {
            'train': 'ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS',
            'validation': 'ID,PRECISION,RECALL,F1,ACCURACY',
            'test': 'ID,PRECISION,RECALL,F1,ACCURACY'
        }

    def _eval(self, data_loaders=None, logger=None, gen_images=False, score_acc=None):
        assert isinstance(score_acc, ScoreAccumulator)
        with torch.no_grad():
            for loader in data_loaders:
                img_obj = loader.dataset.image_objects[0]
                x, y = img_obj.working_arr.shape[0], img_obj.working_arr.shape[1]
                predicted_img = torch.FloatTensor(x, y).fill_(0).to(self.device)
                map_img = torch.FloatTensor(x, y).fill_(0).to(self.device)

                gt = torch.FloatTensor(img_obj.ground_truth).to(self.device)

                for i, data in enumerate(loader, 1):
                    inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).float()
                    clip_ix = data['clip_ix'].to(self.device).int()

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predicted_map = outputs[:, 1, :, :]

                    for j in range(predicted_map.shape[0]):
                        p, q, r, s = clip_ix[j]
                        predicted_img[p:q, r:s] = predicted[j]
                        map_img[p:q, r:s] = predicted_map[j]
                    print('Batch: ', i, end='\r')

                img_score = ScoreAccumulator()
                map_img = torch.exp(map_img) * 255
                predicted_img = predicted_img * 255

                if gen_images:
                    map_img = map_img.cpu().numpy()
                    predicted_img = predicted_img.cpu().numpy()
                    img_score.add_array(predicted_img, img_obj.ground_truth)
                    self.conf['acc'].accumulate(img_score)  # Global score

                    IMG.fromarray(np.array(predicted_img, dtype=np.uint8)).save(
                        os.path.join(self.log_dir, 'pred_' + img_obj.file_name.split('.')[0] + '.png'))
                    IMG.fromarray(np.array(map_img, dtype=np.uint8)).save(
                        os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + '.png'))
                else:
                    img_score.add_tensor(predicted_img, gt)
                    score_acc.accumulate(img_score)

                prf1a = img_score.get_prfa()
                print(img_obj.file_name, ' PRF1A', prf1a)
                self.flush(logger, ','.join(str(x) for x in [img_obj.file_name] + prf1a))
