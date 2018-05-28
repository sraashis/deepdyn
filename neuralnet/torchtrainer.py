import os
from itertools import count
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.autograd import Variable

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
              validationloader=None):

        if validationloader is None:
            raise ValueError('Please provide validation loader.')

        self.model.train()
        self.model.cuda() if use_gpu else self.model.cpu()
        print('Training...')
        for epoch in range(0, epochs):
            running_loss = 0.0
            accumulated_labels = []
            accumulated_predictions = []
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
                p, r, f1, s = self.get_score(labels.numpy().squeeze().ravel(), predicted.numpy().squeeze().ravel())

                # Accumulate to calculate score of log frequency batches for better logging
                accumulated_predictions += predicted.numpy().tolist()
                accumulated_labels += labels.numpy().tolist()

                ################## Tensorboard logger setup ###############
                ###########################################################
                if self.to_tenserboard:
                    step = next(self.res['train_counter'])
                    self.logger.scalar_summary('loss/training', current_loss, step)
                    self.logger.scalar_summary('F1/training', f1, step)
                    self.logger.scalar_summary('precision-recall/training', r, p)
                    self.logger.scalar_summary('Support/training', s, step)

                    for tag, value in self.model.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, value.data.cpu().numpy(), step)
                        self.logger.histo_summary(tag + '/gradients', value.grad.data.cpu().numpy(), step)

                    images_to_tb = inputs.view(-1, dataloader.dataset.img_width, dataloader.dataset.img_height)[
                                   :10].data.cpu().numpy()
                    self.logger.image_summary('images/training', images_to_tb, step)
                ###### Tensorboard logger END ##############################
                ############################################################

                if (i + 1) % log_frequency == 0:  # Inspect the loss of every log_frequency batches
                    current_loss = running_loss / log_frequency if (i + 1) % log_frequency == 0 \
                        else (i + 1) % log_frequency
                    p, r, f1, s = self.get_score(np.array(accumulated_labels).ravel(),
                                                   np.array(accumulated_predictions).ravel())
                    running_loss = 0.0
                    accumulated_labels = []
                    accumulated_predictions = []

                print('[Epochs:%d/%d Batches:%d/%d, loss:%.3f] pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (epoch + 1, epochs, i + 1, dataloader.__len__(), current_loss, p, r, f1, s),
                      end='\r' if running_loss > 0 else '\n')

            self.checkpoint['epochs'] += 1
            self.evaluate(dataloader=validationloader, use_gpu=use_gpu, force_checkpoint=False, save_best=True)

    def evaluate(self, dataloader=None, use_gpu=False, force_checkpoint=False, save_best=False):

        self.model.eval()
        self.model.cuda() if use_gpu else self.model.cpu()

        all_predictions = []
        all_labels = []
        print('\nEvaluating...')

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.cuda() if use_gpu else inputs.cpu()
            labels = labels.cuda() if use_gpu else labels.cpu()

            outputs = self.model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)

            # Save scores
            all_predictions += predicted.numpy().tolist()
            all_labels += labels.numpy().tolist()

            p, r, f1, s = self.get_score(labels.numpy().squeeze().ravel(), predicted.numpy().squeeze().ravel())
            print('Batch[%d/%d] pre:%.3f rec:%.3f f1:%.3f acc:%.3f' % (
                i + 1, dataloader.__len__(), p, r, f1, s),
                  end='\r')

            ########## Feeding to tensorboard starts here...#####################
            ####################################################################
            if self.to_tenserboard:
                step = next(self.res['val_counter'])
                self.logger.scalar_summary('F1/validation', f1, step)
                self.logger.scalar_summary('precision-recall/validation', r, p)
                self.logger.scalar_summary('Acc/validation', s, step)
            #### Tensorfeed stops here# #########################################
            #####################################################################

        print()
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        p, r, f1, s = self.get_score(all_labels.ravel(), all_predictions.ravel())
        print('[FINAL ::: Precision:%.3f Recall:%.3f F1:%.3f Acc:%.3f]' % (p, r, f1, s))
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

        if score > self.checkpoint['score']:
            print('Score improved from ',
                  str(self.checkpoint['score']) + ' to ' + str(score) + '. Saving model..')
            self._save_checkpoint(
                NNTrainer._checkpoint(epochs=self.checkpoint['epochs'], model=self.model,
                                      score=score))
        else:
            self._save_checkpoint(self.checkpoint)
            print('Score did not improve. _was:' + str(self.checkpoint['score']))

    def get_score(self, y_true, y_pred):
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average='binary')
        p = 0.0 if p is None else p
        r = 0.0 if r is None else r
        f1 = 0.0 if f1 is None else f1
        s = accuracy_score(y_true, y_pred)
        return round(p, 3), round(r, 3), round(f1, 3), round(s, 3)
