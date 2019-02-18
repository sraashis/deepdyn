"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os

import numpy as np
import math
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
        self.dparm = self.run_conf.get("Funcs").get('dparm')

    def train(self, optimizer=None, data_loader=None, validation_loader=None):

        if validation_loader is None:
            raise ValueError('Please provide validation loader.')

        logger = NNTrainer.get_logger(self.train_log_file,
                                      header='ID,EPOCH,BATCH,LOSS')

        val_logger = NNTrainer.get_logger(self.validation_log_file,
                                          header='ID,LOSS')

        print('Training...')
        for epoch in range(1, self.epochs + 1):

            self.model.train()
            running_loss = 0.0
            self._adjust_learning_rate(optimizer=optimizer, epoch=epoch)
            self.checkpoint['total_epochs'] = epoch

            for i, data in enumerate(data_loader, 1):
                inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).float()

                optimizer.zero_grad()
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                loss = F.mse_loss(outputs.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                running_loss += current_loss

                if i % self.log_frequency == 0:
                    print('Epochs[%d/%d] Batch[%d/%d] MSE loss:%.5f ' %
                          (
                              epoch, self.epochs, i, data_loader.__len__(), running_loss / self.log_frequency))
                    running_loss = 0.0

                self.flush(logger, ','.join(str(x) for x in [0, epoch, i, current_loss]))

            self.plot_train(file=self.train_log_file, batches_per_epochs=data_loader.__len__(), keys=['LOSS'])
            if epoch % self.validation_frequency == 0:
                self.evaluate(data_loaders=validation_loader, logger=val_logger, gen_images=False)
                if self.early_stop(patience=75):
                    return

            self.plot_val(self.validation_log_file, batches_per_epoch=len(validation_loader))

        try:
            logger.close()
            val_logger.close()
        except IOError:
            pass

    def evaluate(self, data_loaders=None, logger=None, gen_images=False):
        assert (logger is not None), 'Please Provide a logger'
        self.model.eval()

        eval_loss = 0.0
        print('\nEvaluating...')
        with torch.no_grad():
            for loader in data_loaders:
                img_obj = loader.dataset.image_objects[0]
                x, y = img_obj.working_arr.shape[0], img_obj.working_arr.shape[1]
                predicted_img = torch.FloatTensor(x, y).fill_(0).to(self.device)
                map_img = torch.FloatTensor(x, y).fill_(0).to(self.device)

                img_loss = 0.0
                for i, data in enumerate(loader, 1):
                    inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).float()
                    clip_ix = data['clip_ix'].to(self.device).int()

                    outputs = self.model(inputs)
                    loss = F.mse_loss(outputs.squeeze(), labels.squeeze()).item()

                    img_loss += loss

                    for j in range(outputs.shape[0]):
                        p, q, r, s = clip_ix[j]
                        map_img[p:q, r:s] = outputs[j]

                    print('Batch: ', i, end='\r')

                if gen_images:
                    map_img = map_img.cpu().numpy()
                    IMG.fromarray(np.array(map_img, dtype=np.uint8)).save(
                        os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + '.png'))
                else:
                    eval_loss += img_loss / loader.__len__()
                print('\n' + img_obj.file_name, ' Image LOSS: ', img_loss / loader.__len__())

                self.flush(logger, ','.join(str(x) for x in [img_obj.file_name, img_loss / loader.__len__()]))

        self._save_if_better(score=len(data_loaders)/eval_loss)
