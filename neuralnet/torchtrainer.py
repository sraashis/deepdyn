import os
from time import time

import torch
import torch.nn.functional as F

from neuralnet.utils.measurements import ScoreAccumulator


class NNTrainer:
    def __init__(self, model=None, checkpoint_file='{}'.format(time()) + '.tar',
                 log_file='{}'.format(time()) + '.csv',
                 use_gpu=True):
        if torch.cuda.is_available():
            self.device = torch.device("cuda" if use_gpu else "cpu")
        else:
            print('### GPU not found.')
            self.device = torch.device("cpu")
        self.model = model.to(self.device)

        os.makedirs('net_logs', exist_ok=True)
        self.logger = self.get_logger(log_file)
        self.checkpoint_file = os.path.join('net_logs', checkpoint_file)
        self.checkpoint = {'epochs': 0, 'state': None, 'score': 0.0, 'model': 'EMPTY'}

    def train(self, optimizer=None, data_loader=None, epochs=None, log_frequency=200,
              validation_loader=None, force_checkpoint=False):

        if validation_loader is None:
            raise ValueError('Please provide validation loader.')

        print('Training...')
        for epoch in range(0, epochs):
            self.model.train()
            score_acc = ScoreAccumulator()
            running_loss = 0.0
            self.adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1)
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data[-2].to(self.device), data[-1].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.nll_loss(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                current_loss = loss.item()
                _, predicted = torch.max(outputs, 1)

                p, r, f1, a = score_acc.reset().add(labels, predicted).get_prf1a()
                if (i + 1) % log_frequency == 0:  # Inspect the loss of every log_frequency batches
                    current_loss = running_loss / log_frequency if (i + 1) % log_frequency == 0 \
                        else (i + 1) % log_frequency
                    running_loss = 0.0

                self.flush(self.logger, ','.join(str(x) for x in [0, 0, epoch + 1, i + 1, p, r, f1, a, current_loss]))
                print('Epochs[%d/%d] Batch[%d/%d] loss:%.5f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (epoch + 1, epochs, i + 1, data_loader.__len__(), current_loss, p, r, f1, a),
                      end='\r' if running_loss > 0 else '\n')

            self.checkpoint['epochs'] += 1
            self.evaluate(data_loader=validation_loader, force_checkpoint=force_checkpoint,
                          mode='train', logger=self.logger)
        try:
            self.logger.close()
        except IOError:
            pass

    def evaluate(self, data_loader=None, force_checkpoint=False, mode='val', logger=None, **kwargs):

        assert (logger is not None), 'Please Provide a logger'
        assert (mode == 'val' or mode == 'train'), 'Mode can either be val or train'
        self.model.eval()
        print('\nEvaluating...')
        with torch.no_grad():
            return self._evaluate(data_loader=data_loader, force_checkpoint=force_checkpoint, mode=mode, logger=logger)

    def _evaluate(self, data_loader=None, force_checkpoint=None, mode=None, logger=None):
        raise NotImplementedError('ERROR!!!!! Must be implemented')

    def _save_checkpoint(self, checkpoint):
        torch.save(checkpoint, self.checkpoint_file)
        self.checkpoint = checkpoint

    @staticmethod
    def _checkpoint(epochs=None, model=None, score=None):
        return {'state': model.state_dict(),
                'epochs': epochs,
                'score': score,
                'model': str(model)}

    def resume_from_checkpoint(self, parallel_trained=False):
        try:
            self.checkpoint = torch.load(self.checkpoint_file)
            if parallel_trained:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in self.checkpoint['state'].items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(self.checkpoint['state'])
            print('Resumed last checkpoint: ' + self.checkpoint_file)
        except Exception as e:
            print('ERROR: ' + str(e))

    def _save_if_better(self, force_checkpoint=None, score=None):
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
            print('Score did not improve. _was:' + str(self.checkpoint['score']))

    @staticmethod
    def get_logger(log_file):
        file = open(os.path.join('net_logs', log_file), 'w')
        NNTrainer.flush(file, 'ID,TYPE,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS')
        return file

    @staticmethod
    def flush(logger, msg):
        if logger is not None:
            logger.write(msg + '\n')
            logger.flush()

    @staticmethod
    def adjust_learning_rate(optimizer, epoch):
        if epoch % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.50
