import os

import PIL.Image as IMG
import numpy as np
import torch
import torch.nn.functional as F

import utils.img_utils as imgutils
from neuralnet.torchtrainer import NNTrainer
from neuralnet.utils.measurements import ScoreAccumulator

sep = os.sep


class ThrnetTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)

    def train(self, optimizer=None, data_loader=None, epochs=None, log_frequency=200,
              validation_loader=None, force_checkpoint=False):

        if validation_loader is None:
            raise ValueError('Please provide validation loader.')
        logger = self.get_logger(self.log_file)
        print('Training...')
        for epoch in range(0, epochs):
            self.model.train()
            score_acc = ScoreAccumulator()
            running_loss = 0.0
            self.adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1)
            for i, data in enumerate(data_loader, 0):

                inputs, _, ths = data[1].to(self.device), data[2].to(self.device), data[3].to(self.device)
                optimizer.zero_grad()

                outputs = self.model(inputs).squeeze()
                # loss = dice_loss.forward(predicted, Variable(labels, requires_grad=True))
                # print(outputs.type(), labels.type(), inputs.type())
                loss = F.mse_loss(outputs, ths.type(torch.FloatTensor).to(self.device))
                loss.backward()
                optimizer.step()
                # print(outputs, loss)

                running_loss += float(loss.item())
                current_loss = loss.item()
                if (i + 1) % log_frequency == 0:  # Inspect the loss of every log_frequency batches
                    current_loss = running_loss / log_frequency if (i + 1) % log_frequency == 0 \
                        else (i + 1) % log_frequency
                    running_loss = 0.0

                self.flush(logger, ','.join(str(x) for x in [0, 0, epoch + 1, i + 1, 0, 0, 0, 0, current_loss]))
                print('Epochs[%d/%d] Batch[%d/%d] loss:%.5f' %
                      (epoch + 1, epochs, i + 1, data_loader.__len__(), current_loss),
                      end='\r' if running_loss > 0 else '\n')

            self.checkpoint['epochs'] += 1
            self.evaluate(data_loader=validation_loader, force_checkpoint=force_checkpoint,
                          mode='train', logger=logger)
        try:
            logger.close()
        except IOError:
            pass

    def evaluate(self, data_loader=None, force_checkpoint=False, mode=None, logger=None, **kwargs):

        assert (mode == 'eval' or mode == 'train'), 'Mode can either be eval or train'
        assert (logger is not None), 'Please Provide a logger'
        to_dir = kwargs['segmented_out'] if 'segmented_out' in kwargs else None
        patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else None

        self.model.eval()
        data_loader = data_loader if isinstance(data_loader, list) else [data_loader]
        score_acc = ScoreAccumulator()
        print('\nEvaluating...')
        with torch.no_grad():
            for loader in data_loader:
                if mode is 'train':
                    score_acc.accumulate(self._evaluate(data_loader=loader,
                                                        force_checkpoint=force_checkpoint, mode=mode, logger=logger))
                if mode is 'eval' and to_dir is not None:
                    predictions = self._evaluate(data_loader=loader,
                                                 force_checkpoint=force_checkpoint,
                                                 mode=mode,
                                                 logger=logger)
                    segmented = imgutils.merge_patches(patches=predictions * 255,
                                                       image_size=loader.dataset.image_objects[0].working_arr.shape,
                                                       patch_size=patch_size)

                    IMG.fromarray(segmented).save(to_dir + sep + loader.dataset.image_objects[0].file_name + '.png')
        if mode is 'train':
            self._save_if_better(force_checkpoint=force_checkpoint, score=score_acc.get_prf1a()[2])

    def _evaluate(self, data_loader=None, force_checkpoint=False, mode=None, logger=None):

        assert (mode == 'eval' or mode == 'train'), 'Mode can either be eval or train'
        assert (logger is not None), 'Please Provide a logger'
        score_acc = ScoreAccumulator()
        all_predictions = []
        for i, data in enumerate(data_loader, 0):
            ID, inputs, labels, thr_y = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[
                3].to(self.device)
            thr = self.model(inputs)

            inputs = inputs.squeeze() * 255
            thr = thr.squeeze()
            # print(input_img.type(), thr.type())
            # print(input_img.shape, thr.shape)
            for o in range(inputs.shape[0]):
                inputs[o, :, :][inputs[o, :, :] > thr[o]] = 255
                inputs[o, :, :][inputs[o, :, :] <= thr[o]] = 0
            # Accumulate scores
            if mode is 'eval':
                all_predictions += inputs.clone().cpu().numpy().tolist()
            p, r, f1, a = score_acc.add(labels, inputs).get_prf1a()
            loss = F.mse_loss(thr, thr_y.type(torch.FloatTensor).to(self.device))
            print('Batch[%d/%d] pre:%.3f rec:%.3f f1:%.3f acc:%.3f MSE:%.5f' % (
                i + 1, data_loader.__len__(), p, r, f1, a, loss),
                  end='\r')
        self.flush(logger, ','.join(str(x) for x in
                                    [data_loader.dataset.image_objects[0].file_name, 1, self.checkpoint['epochs'],
                                     0] + score_acc.get_prf1a()))
        print()
        if mode is 'eval':
            all_predictions = np.array(all_predictions)
        if mode is 'train':
            return score_acc
        return all_predictions
