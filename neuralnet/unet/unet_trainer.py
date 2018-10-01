"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as IMG

from neuralnet.torchtrainer import NNTrainer
from neuralnet.utils.measurements import ScoreAccumulator

sep = os.sep


class UNetNNTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')

    def train(self, optimizer=None, data_loader=None, validation_loader=None):

        if validation_loader is None:
            raise ValueError('Please provide validation loader.')

        logger = NNTrainer.get_logger(self.log_file, header='ID,TYPE,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS')
        print('Training...')
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            score_acc = ScoreAccumulator()
            running_loss = 0.0
            self.adjust_learning_rate(optimizer=optimizer, epoch=epoch)
            for i, data in enumerate(data_loader, 1):
                inputs, labels = data['inputs'].to(self.device), data['labels'].long().to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                weights = torch.FloatTensor([random.uniform(1, 100), random.uniform(1, 100)])
                loss = F.nll_loss(outputs, labels, weight=weights.to(self.device))
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                running_loss += current_loss
                p, r, f1, a = score_acc.reset().add_tensor(labels, predicted).get_prf1a()
                if i % self.log_frequency == 0:
                    print('Epochs[%d/%d] Batch[%d/%d] loss:%.5f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                          (
                              epoch, self.epochs, i, data_loader.__len__(), running_loss / self.log_frequency, p, r, f1,
                              a))
                    running_loss = 0.0

                self.flush(logger, ','.join(str(x) for x in [0, 0, epoch, i, p, r, f1, a, current_loss]))

            if epoch % self.validation_frequency == 0:
                self.evaluate(data_loaders=validation_loader, logger=logger, gen_images=False)
        try:
            logger.close()
        except IOError:
            pass

    def evaluate(self, data_loaders=None, logger=None, gen_images=False):
        assert (logger is not None), 'Please Provide a logger'
        self.model.eval()

        print('\nEvaluating...')
        with torch.no_grad():
            eval_score = 0.0

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
                    img_score.add_array(img_obj.ground_truth, predicted_img)
                    IMG.fromarray(np.array(map_img, dtype=np.uint8)).save(
                        os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + '.png'))
                else:
                    img_score.add_tensor(predicted_img, gt)
                    eval_score += img_score.get_prf1a()[2]

                prf1a = img_score.get_prf1a()
                print(img_obj.file_name, ' PRF1A', prf1a)
                self.flush(logger, ','.join(str(x) for x in [img_obj.file_name, 1, 0, 0] + prf1a))

        self._save_if_better(score=eval_score / len(data_loaders))
