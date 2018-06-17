import os
from time import time

import torch
import torch.nn.functional as F

import neuralnet.utils.measurements as mggmt


class NNTrainer:
    def __init__(self, model=None, checkpoint_dir=None, checkpoint_file=None, log_to_file=True, use_gpu=True):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = "{}".format(time()) + checkpoint_file
        self.checkpoint = {'epochs': 0, 'state': None, 'score': 0.0, 'model': 'EMPTY'}
        self.logger = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda" if use_gpu else "cpu")
        else:
            print('### GPU not found.')
            self.device = torch.device("cpu")
        self.model = model.to(self.device)

        if log_to_file:
            self.logger = open(
                os.path.join(self.checkpoint_dir, self.checkpoint_file + '-LOG' + '.csv'), 'w')
            self.logger.write('TYPE,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS\n')

    def train(self, optimizer=None, dataloader=None, epochs=None, log_frequency=200,
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

        print('Training...')
        TP, FP, TN, FN = [0] * 4
        for epoch in range(0, epochs):
            running_loss = 0.0
            self.model.train()
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                current_loss = loss.item()
                _, predicted = torch.max(outputs, 1)

                _tp, _fp, _tn, _fn = self.get_score(labels, predicted)
                TP += _tp
                TN += _tn
                FP += _fp
                FN += _fn
                p, r, f1, a = mggmt.get_prf1a(TP, FP, TN, FN)

                if (i + 1) % log_frequency == 0:  # Inspect the loss of every log_frequency batches
                    current_loss = running_loss / log_frequency if (i + 1) % log_frequency == 0 \
                        else (i + 1) % log_frequency
                    running_loss = 0.0

                self._log(','.join(str(x) for x in [0, epoch + 1, i + 1, p, r, f1, a, current_loss]))
                print('Epochs[%d/%d] Batch[%d/%d] loss:%.4f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (epoch + 1, epochs, i + 1, dataloader.__len__(), current_loss, p, r, f1, a),
                      end='\r' if running_loss > 0 else '\n')

            self.checkpoint['epochs'] += 1
            self.evaluate(dataloader=validationloader, force_checkpoint=force_checkpoint,
                          save_best=save_best)

    def evaluate(self, dataloader=None, force_checkpoint=False, save_best=False):
        self.model.eval()
        print('\nEvaluating...')
        with torch.no_grad():
            return self._evaluate(dataloader=dataloader, force_checkpoint=force_checkpoint,
                                  save_best=save_best)

    def _evaluate(self, dataloader=None, force_checkpoint=False, save_best=False):
        raise NotImplementedError('ERROR!!!!! Must be implemented')

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
