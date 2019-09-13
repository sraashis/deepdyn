"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os

import numpy as np
import torch
from PIL import Image as IMG

from torchtrainer.torchtrainer import NNTrainer
from utils.measurements import ScoreAccumulator

sep = os.sep


class MiniUNetTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)
        self.patch_shape = self.conf.get('Params').get('patch_shape')
        self.patch_offset = self.conf.get('Params').get('patch_offset')

    def get_log_headers(self):
        return {
            'train': 'ID,EPOCH,BATCH,LOSS',
            'validation': 'ID,LOSS',
            'test': 'ID,LOSS'
        }

    def _on_epoch_end(self, **kw):
        self.plot_column_keys(file=kw['log_file'], batches_per_epoch=kw['data_loader'].__len__(),
                              keys=['LOSS'])

    def _on_validation_end(self, **kw):
        self.plot_column_keys(file=kw['log_file'], batches_per_epoch=kw['data_loader'].__len__(),
                              keys=['LOSS'])

    def _on_test_end(self, **kw):
        self.plot_column_keys(file=kw['log_file'], batches_per_epoch=1,
                              keys=['F1', 'ACCURACY'])

    # This method takes torch n dataloaders for n image with one image in each and evaluates after training.
    # It is also the base method for both testing and validation

    def evaluate(self, data_loaders=None, logger=None, gen_images=False, score_acc=None):
        assert isinstance(score_acc, ScoreAccumulator)
        for loader in data_loaders:
            img_obj = loader.dataset.image_objects[0]
            x, y = img_obj.working_arr.shape[0], img_obj.working_arr.shape[1]
            predicted_img = torch.FloatTensor(x, y).fill_(0).to(self.device)

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

                print('Batch: ', i, end='\r')

            img_score = ScoreAccumulator()
            if gen_images:  #### Test mode
                predicted_img = predicted_img.cpu().numpy() * 255
                predicted_img[img_obj.extra['fill_in'] == 1] = 255

                img_score.add_array(predicted_img, img_obj.ground_truth)
                ### Only save scores for test images############################

                self.conf['acc'].accumulate(img_score)  # Global score
                prf1a = img_score.get_prfa()
                print(img_obj.file_name, ' PRF1A', prf1a)
                self.flush(logger, ','.join(str(x) for x in [img_obj.file_name] + prf1a))
                #################################################################

                IMG.fromarray(np.array(predicted_img, dtype=np.uint8)).save(
                    os.path.join(self.log_dir, 'pred_' + img_obj.file_name.split('.')[0] + '.png'))
            else:  #### Validation mode
                img_score.add_tensor(predicted_img, gt)
                score_acc.accumulate(img_score)
                prf1a = img_score.get_prfa()
                print(img_obj.file_name, ' PRF1A', prf1a)
                self.flush(logger, ','.join(str(x) for x in [img_obj.file_name] + prf1a))
