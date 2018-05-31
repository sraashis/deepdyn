import os
from itertools import count
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import neuralnet.utils.measurements as mggmt

from neuralnet.utils.tensorboard_logger import Logger


class NNTrainer:
    def __init__(self, model=None, checkpoint_dir=None, checkpoint_file=None, to_tensorboard=False):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file
        self.checkpoint = {'epochs': 0, 'state': None, 'score': 0.0, 'model': 'EMPTY'}
        self.to_tenserboard = to_tensorboard
        self.logger = Logger(log_dir="./logs/{}".format(time()))
        self.res = {'val_counter': count(), 'train_counter': count()}

    def train(self, optimizer=None, dataloader=None, epochs=None, use_gpu=False, log_frequency=200,
              validationloader=None, force_checkpoint=False, save_best=True):

        if validationloader is None:
            raise ValueError('Please provide validation loader.')

        self.model.train()
        self.model.cuda() if use_gpu else self.model.cpu()
        print('Training...')

        TP, FP, TN, FN = [0] * 4
        for epoch in range(0, epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = Variable(inputs.cuda() if use_gpu else inputs.cpu())
                labels = Variable(labels.cuda() if use_gpu else labels.cpu())

                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = F.cross_entropy(outputs, labels)
                loss.cuda() if use_gpu else loss.cpu()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                current_loss = loss.item()

                _, predicted = torch.max(outputs, 1)

                _tp, _fp, _tn, _fn = mggmt.get_score(labels.numpy().squeeze().ravel(),
                                                     predicted.numpy().squeeze().ravel())
                TP += _tp
                TN += _tn
                FP += _fp
                FN += _fn
                p, r, f1, a = mggmt.get_prf1a(TP, FP, TN, FN)

                ################## Tensorboard logger setup ###############
                ###########################################################
                if self.to_tenserboard:
                    step = next(self.res['train_counter'])
                    self.logger.scalar_summary('loss/training', current_loss, step)
                    self.logger.scalar_summary('F1/training', f1, step)
                    self.logger.scalar_summary('Accu/training', a, step)

                    for tag, value in self.model.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, value.data.cpu().numpy(), step)
                        self.logger.histo_summary(tag + '/gradients', value.grad.data.cpu().numpy(), step)

                    images_to_tb = inputs.view(-1, dataloader.dataset.num_rows, dataloader.dataset.num_cols)[
                                   :5].data.cpu().numpy()
                    target_to_tb = labels.view(-1, dataloader.dataset.num_rows, dataloader.dataset.num_cols)[
                                   :5].data.cpu().numpy()
                    self.logger.image_summary('images/training', images_to_tb, step)
                    self.logger.image_summary('ground_truth_images/training', target_to_tb, step)
                ###### Tensorboard logger END ##############################
                ############################################################

                if (i + 1) % log_frequency == 0:  # Inspect the loss of every log_frequency batches
                    current_loss = running_loss / log_frequency if (i + 1) % log_frequency == 0 \
                        else (i + 1) % log_frequency
                    p, r, f1, a = mggmt.get_prf1a(TP, FP, TN, FN)

                    running_loss = 0.0
                    accumulated_labels = []
                    accumulated_predictions = []

                print('[Epochs:%d/%d Batches:%d/%d, loss:%.3f] pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (epoch + 1, epochs, i + 1, dataloader.__len__(), current_loss, p, r, f1, a),
                      end='\r' if running_loss > 0 else '\n')

            self.checkpoint['epochs'] += 1
            self.evaluate(dataloader=validationloader, use_gpu=use_gpu, force_checkpoint=force_checkpoint,
                          save_best=save_best)

    def evaluate(self, dataloader=None, use_gpu=False, force_checkpoint=False, save_best=False):

        self.model.eval()
        self.model.cuda() if use_gpu else self.model.cpu()

        print('\nEvaluating...')
        TP, FP, TN, FN = [0] * 4
        all_predictions = []
        all_labels = []

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.cuda() if use_gpu else inputs.cpu()
            labels = labels.cuda() if use_gpu else labels.cpu()

            outputs = self.model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)

            # Accumulate scores
            all_predictions += predicted.numpy().tolist()
            all_labels += labels.numpy().tolist()

            _tp, _fp, _tn, _fn = mggmt.get_score(labels.numpy().squeeze().ravel(),
                                                 predicted.numpy().squeeze().ravel())
            TP += _tp
            TN += _tn
            FP += _fp
            FN += _fn
            p, r, f1, a = mggmt.get_prf1a(TP, FP, TN, FN)

            print('Batch[%d/%d] pre:%.3f rec:%.3f f1:%.3f acc:%.3f' % (
                i + 1, dataloader.__len__(), p, r, f1, a),
                  end='\r')

            ########## Feeding to tensorboard starts here...#####################
            ####################################################################
            if self.to_tenserboard:
                step = next(self.res['val_counter'])
                self.logger.scalar_summary('F1/validation', f1, step)
                self.logger.scalar_summary('Acc/validation', a, step)
            #### Tensorfeed stops here# #########################################
            #####################################################################

        print()
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        self._save_if_better(save_best=save_best, force_checkpoint=force_checkpoint, score=f1)

        return all_predictions, all_labels

    def _save_checkpoint(self, checkpoint):
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, self.checkpoint_file))
        self.checkpoint = checkpoint

    @staticmethod
    def _checkpoint(epochs=None, model=None, score=None):
        return {'state': model.state_dict(),
                'epochs': epochs,
                'score': score,
                'model': str(model)}

    def resume_from_checkpoint(self):
        try:
            self.checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.checkpoint_file))
            self.model.load_state_dict(self.checkpoint['state'])
            print('Resumed last checkpoint: ' + self.checkpoint_file)
        except Exception as e:
            print('ERROR: ' + str(e))

    def _save_if_better(self, save_best=None, force_checkpoint=None, score=None):

        if not save_best:
            return

        if force_checkpoint:
            self._save_checkpoint(
                NNTrainer._checkpoint(epochs=self.checkpoint['epochs'], model=self.model,
                                      score=score))
            print('FORCED checkpoint saved. ')
            return

        if score > self.checkpoint['score']:
            print('Score improved from ',
                  str(self.checkpoint['score']) + ' to ' + str(score) + '. Saving model..')
            self._save_checkpoint(
                NNTrainer._checkpoint(epochs=self.checkpoint['epochs'], model=self.model,
                                      score=score))
        else:
            self._save_checkpoint(self.checkpoint)
            print('Score did not improve. _was:' + str(self.checkpoint['score']))
