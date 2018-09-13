import os

import PIL.Image as IMG
import numpy as np
import torch
import torch.nn.functional as F

import utils.img_utils as imgutils
from neuralnet.torchtrainer import NNTrainer

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
                inputs, y_thresholds = data['inputs'].to(self.device), data['y_thresholds'].to(self.device)

                optimizer.zero_grad()
                thr = self.model(inputs)
                thr = thr.squeeze()
                loss = F.mse_loss(thr, y_thresholds.float())
                loss.backward()
                optimizer.step()

                current_loss = loss.item() / thr.numel()

                running_loss += current_loss
                if i % self.log_frequency == 0:
                    print('Epochs[%d/%d] Batch[%d/%d] mse:%.5f' %
                          (
                              epoch, self.epochs, i, data_loader.__len__(), running_loss / self.log_frequency))
                    running_loss = 0.0

                self.flush(logger, ','.join(str(x) for x in [0, 0, epoch, i, current_loss]))

            if epoch % self.validation_frequency == 0:
                self.evaluate(data_loaders=validation_loader, logger=logger,
                              mode='train')
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
            for loader in data_loaders:
                img_obj = loader.dataset.image_objects[0]
                segmented_img = []
                img_loss = 0.0
                for i, data in enumerate(loader, 1):
                    inputs, labels, y_thr = data['inputs'].to(self.device), data['labels'].to(self.device), data[
                        'y_thresholds'].to(self.device)
                    prob_map = data['prob_map'].to(self.device)

                    thr = self.model(inputs)
                    thr = thr.squeeze()

                    loss = F.mse_loss(thr, y_thr.float().squeeze())
                    current_loss = loss.item() / thr.numel()
                    img_loss += current_loss

                    segmented = (prob_map >= thr[..., None][..., None].byte())
                    if mode is 'test':
                        segmented_img += segmented.clone().cpu().numpy().tolist()

                    self.flush(logger, ','.join(
                        str(x) for x in
                        [img_obj.file_name, 1, self.checkpoint['epochs'], 0] + [current_loss]))

                img_loss = img_loss / loader.__len__()  # Number of batches
                eval_loss += img_loss
                print(img_obj.file_name + ' MSE LOSS: ', img_loss)
                if mode is 'test':
                    segmented_img = np.array(segmented_img, dtype=np.uint8) * 255

                    maps_img = imgutils.merge_patches(patches=segmented_img, image_size=img_obj.working_arr.shape,
                                                      patch_size=self.patch_shape,
                                                      offset_row_col=self.patch_offset)
                    IMG.fromarray(maps_img).save(os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + '.png'))

        if mode is 'train':
            self._save_if_better(score=1 / (eval_loss / len(data_loaders)))
