import os
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import neuralnet.utils.measurements as mggmt


class NNTrainer:
    def __init__(self, model=None, checkpoint_dir=None, checkpoint_file=None, log_to_file=True):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file
        self.checkpoint = {'epochs': 0, 'state': None, 'score': 0.0, 'model': 'EMPTY'}
        self.logger = None

        if log_to_file:
            self.logger = open(
                os.path.join(self.checkpoint_dir, checkpoint_file + 'LOG-' + "{}".format(time())) + '.csv', 'w')
            self.logger.write('TYPE,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY\n')

    def train(self, optimizer=None, dataloader=None, epochs=None, use_gpu=None, log_frequency=200,
              validationloader=None, force_checkpoint=False, save_best=True):

        """
        :param optimizer:
        :param dataloader:
        :param epochs:
        :param use_gpu: (0, 1, None)
        :param log_frequency:
        :param validationloader:
        :param force_checkpoint:
        :param save_best:
        :return:
        """

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
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]
                current_loss = loss.data[0]

                _, predicted = torch.max(outputs, 1)

                _tp, _fp, _tn, _fn = self.get_score(labels.data, predicted.data)
                TP += _tp
                TN += _tn
                FP += _fp
                FN += _fn
                p, r, f1, a = mggmt.get_prf1a(TP, FP, TN, FN)

                if (i + 1) % log_frequency == 0:  # Inspect the loss of every log_frequency batches
                    current_loss = running_loss / log_frequency if (i + 1) % log_frequency == 0 \
                        else (i + 1) % log_frequency
                    running_loss = 0.0

                self._log(','.join(str(x) for x in [0, epoch + 1, i + 1, p, r, a, f1]))
                print('Epochs[%d/%d] Batch[%d/%d] loss:%.3f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (epoch + 1, epochs, i + 1, dataloader.__len__(), current_loss, p, r, f1, a),
                      end='\r' if running_loss > 0 else '\n')

            self.checkpoint['epochs'] += 1
            self.evaluate(dataloader=validationloader, use_gpu=use_gpu, force_checkpoint=force_checkpoint,
                          save_best=save_best)

    def evaluate(self, dataloader=None, use_gpu=False, force_checkpoint=False, save_best=False):

        self.model.eval()
        self.model.cuda() if use_gpu else self.model.cpu()

        TP, FP, TN, FN = [0] * 4
        all_predictions = []
        all_labels = []

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.cuda() if use_gpu else inputs.cpu()
            labels = labels.cuda() if use_gpu else labels.cpu()

            outputs = self.model(Variable(inputs))
            _, predicted = torch.max(outputs.data[0], 1)

            # Accumulate scores
            all_predictions += predicted.clone().cpu().numpy().tolist()
            all_labels += labels.data.clone().cpu().numpy().tolist()

            _tp, _fp, _tn, _fn = self.get_score(labels, predicted)

            TP += _tp
            TN += _tn
            FP += _fp
            FN += _fn
            p, r, f1, a = mggmt.get_prf1a(TP, FP, TN, FN)

            self._log(','.join(str(x) for x in [1, 0, i + 1, p, r, a, f1]))
            print('Evaluating Batch[%d/%d] pre:%.3f rec:%.3f f1:%.3f acc:%.3f' % (
                i + 1, dataloader.__len__(), p, r, f1, a),
                  end='\r')

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

    def _log(self, msg):
        if self.logger is not None:
            self.logger.write(msg + '\n')
            self.logger.flush()

    def get_score(self, y_true_tensor, y_pred_tensor):
        TP, FP, TN, FN = [0] * 4
        y_true = y_true_tensor.clone().cpu().numpy().squeeze().ravel()
        y_pred = y_pred_tensor.clone().cpu().numpy().squeeze().ravel()
        for i in range(len(y_pred)):
            if y_true[i] == y_pred[i] == 1:
                TP += 1
            if y_pred[i] == 1 and y_true[i] != y_pred[i]:
                FP += 1
            if y_true[i] == y_pred[i] == 0:
                TN += 1
            if y_pred[i] == 0 and y_true[i] != y_pred[i]:
                FN += 1
        return TP, FP, TN, FN
