import os
import random

import torch
import torch.nn.functional as F

from neuralnet.torchtrainer import NNTrainer
from neuralnet.utils.measurements import ScoreAccumulator

sep = os.sep


class ThrnetTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')

    # FOR THRESHOLD BASED NETWORK
    # def train(self, optimizer=None, data_loader=None, validation_loader=None):
    #
    #     if validation_loader is None:
    #         raise ValueError('Please provide validation loader.')
    #
    #     logger = NNTrainer.get_logger(self.log_file, 'ID,TYPE,EPOCH,BATCH,LOSS')
    #     print('Training...')
    #     for epoch in range(1, self.epochs + 1):
    #         self.model.train()
    #         running_loss = 0.0
    #         self.adjust_learning_rate(optimizer=optimizer, epoch=epoch)
    #         for i, data in enumerate(data_loader, 1):
    #             inputs, y_thresholds = data['inputs'].to(self.device), data['y_thresholds'].float().to(self.device)
    #
    #             optimizer.zero_grad()
    #             thr_map = self.model(inputs)
    #
    #             # if True:
    #             #     print(torch.cat([y_thresholds[..., None], thr_map], 1))
    #             #     print('-------------------------------------------------')
    #
    #             y_thresholds = y_thresholds.squeeze()
    #             thr_map = thr_map.squeeze()
    #             loss = F.mse_loss(thr_map, y_thresholds)
    #             loss.backward()
    #             optimizer.step()
    #
    #             current_loss = math.sqrt(loss.item())
    #             running_loss += current_loss
    #             if i % self.log_frequency == 0:
    #                 print('Epochs[%d/%d] Batch[%d/%d] mse:%.5f' %
    #                       (
    #                           epoch, self.epochs, i, data_loader.__len__(), running_loss / self.log_frequency))
    #                 running_loss = 0.0
    #
    #             self.flush(logger, ','.join(str(x) for x in [0, 0, epoch, i, current_loss]))
    #
    #         if epoch % self.validation_frequency == 0:
    #             self.evaluate(data_loaders=validation_loader, logger=logger,
    #                           mode='test')
    #     try:
    #         logger.close()
    #     except IOError:
    #         pass

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

                weights = torch.FloatTensor([random.uniform(1, 20), random.uniform(1, 20)])
                loss = F.nll_loss(outputs, labels)
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
