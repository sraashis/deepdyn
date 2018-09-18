import math
import os

import PIL.Image as IMG
import numpy as np
import torch
import torch.nn.functional as F

from neuralnet.torchtrainer import NNTrainer
from neuralnet.utils.measurements import ScoreAccumulator
from utils import img_utils as iu

sep = os.sep


class ThrnetTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')

    def train(self, optimizer=None, data_loader=None, validation_loader=None):

        if validation_loader is None:
            raise ValueError('Please provide validation loader.')

        logger = NNTrainer.get_logger(self.log_file, 'ID,TYPE,EPOCH,BATCH,LOSS')
        print('Training...')
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            self.adjust_learning_rate(optimizer=optimizer, epoch=epoch)
            for i, data in enumerate(data_loader, 1):
                inputs, y_thresholds = data['inputs'].to(self.device), data['y_thresholds'].float().to(self.device)

                optimizer.zero_grad()
                thr_map = self.model(inputs)

                if True:
                    print(torch.cat([y_thresholds[..., None], thr_map], 1))
                    print('-------------------------------------------------')

                y_thresholds = y_thresholds.squeeze()
                thr_map = thr_map.squeeze()
                loss = F.mse_loss(thr_map, y_thresholds)
                loss.backward()
                optimizer.step()

                current_loss = math.sqrt(loss.item())
                running_loss += current_loss
                if i % self.log_frequency == 0:
                    print('Epochs[%d/%d] Batch[%d/%d] mse:%.5f' %
                          (
                              epoch, self.epochs, i, data_loader.__len__(), running_loss / self.log_frequency))
                    running_loss = 0.0

                self.flush(logger, ','.join(str(x) for x in [0, 0, epoch, i, current_loss]))

            if epoch % self.validation_frequency == 0:
                self.evaluate(data_loaders=validation_loader, logger=logger,
                              mode='test')
        try:
            logger.close()
        except IOError:
            pass

    def evaluate(self, data_loaders=None, logger=None, mode=None):
        assert (logger is not None), 'Please Provide a logger'
        self.model.eval()

        print('\nEvaluating...')
        with torch.no_grad():
            eval_loss = 0.0
            eval_score = ScoreAccumulator()
            for loader in data_loaders:
                img_obj = loader.dataset.image_objects[0]
                segmented_img = np.zeros_like(img_obj.working_arr)
                img_loss = 0.0
                for i, data in enumerate(loader, 1):
                    inputs = data['inputs'].to(self.device)
                    prob_map = data['prob_map'].to(self.device)
                    y_thresholds = data['y_thresholds'].float().to(self.device)
                    clip_ix = data['clip_ix'].int().squeeze().to(self.device)
                    thr_map = self.model(inputs)
                    truth = data['truth'].squeeze().to(self.device)

                    # if True:
                    #     print(torch.cat([y_thresholds[..., None], thr_map], 1))
                    #     print('-------------------------------------------------')

                    thr_map = thr_map.squeeze()
                    prob_map = prob_map.squeeze()
                    y_thresholds = y_thresholds.squeeze()

                    loss = F.mse_loss(thr_map, y_thresholds)
                    current_loss = math.sqrt(loss.item())
                    img_loss += current_loss

                    thr = thr_map.abs_()[..., None][..., None]
                    segmented = (prob_map >= thr.byte())

                    # batch_score = ScoreAccumulator().add_tensor(segmented, truth)
                    print('Loss: ', current_loss, end='\r')
                    self.flush(logger, ','.join(
                        str(x) for x in
                        [img_obj.file_name, 1, self.checkpoint['epochs'], 0] + [current_loss]))

                    if mode == 'test':
                        for j, patch in enumerate(segmented):
                            p, q, r, s = clip_ix[j]
                            segmented_img[p:q, r:s] += patch.cpu().numpy().squeeze()

                img_loss = img_loss / loader.__len__()  # Number of batches
                eval_loss += img_loss
                if mode is 'test':
                    segmented_img[segmented_img > 0] = 255
                    img_score = ScoreAccumulator()
                    img_score.add_array(segmented_img, img_obj.ground_truth)
                    eval_score.accumulate(img_score)
                    IMG.fromarray(segmented_img).save(
                        os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + '.png'))

                    print('\n' + img_obj.file_name + ' prf1a: ', img_score.get_prf1a())

        if mode is 'train' or True:
            self._save_if_better(score=eval_score.get_prf1a()[2])
