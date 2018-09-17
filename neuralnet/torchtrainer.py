"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os
import random
import sys

import torch
import torch.nn.functional as F

from neuralnet.utils.measurements import ScoreAccumulator


class NNTrainer:

    def __init__(self, run_conf=None, model=None):

        self.run_conf = run_conf
        self.log_dir = self.run_conf.get('Dirs').get('logs', 'net_logs')
        self.use_gpu = self.run_conf['Params'].get('use_gpu', False)
        self.epochs = self.run_conf.get('Params').get('epochs', 100)
        self.log_frequency = self.run_conf.get('Params').get('log_frequency', 10)
        self.validation_frequency = self.run_conf.get('Params').get('validation_frequency', 1)

        self.checkpoint_file = os.path.join(self.log_dir, self.run_conf.get('Params').get('checkpoint_file'))
        self.temp_chk_file = os.path.join(self.log_dir, 'RUNNING' + self.run_conf.get('Params').get('checkpoint_file'))
        self.log_file = os.path.join(self.log_dir, self.run_conf.get('Params').get('checkpoint_file') + '-TRAIN.csv')

        if torch.cuda.is_available():
            self.device = torch.device("cuda" if self.use_gpu else "cpu")
        else:
            print('### GPU not found.')
            self.device = torch.device("cpu")

        self.model = model.to(self.device)

        self.checkpoint = {'epochs': 0, 'state': None, 'score': 0.0, 'model': 'EMPTY'}

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
                inputs, labels = data['inputs'].to(self.device), data['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                weights = torch.FloatTensor([random.uniform(1, 100), random.uniform(1, 100)])
                loss = F.nll_loss(outputs, labels, weight=weights.to(self.device))
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
                self.evaluate(data_loaders=validation_loader, logger=logger,
                              mode='train')
        try:
            logger.close()
        except IOError:
            pass

    def evaluate(self, data_loaders=None, logger=None, mode=None):
        raise NotImplementedError('ERROR!!!!! Must be implemented')

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
            print('RESUMED FROM CHECKPOINT: ' + self.checkpoint_file)
        except Exception as e:
            print('ERROR: ' + str(e))

    def _save_if_better(self, score=None):
        score = round(score, 5)
        current_epoch = self.checkpoint['epochs'] + self.validation_frequency
        current_chk = {'state': self.model.state_dict(),
                       'epochs': current_epoch,
                       'score': score,
                       'model': str(self.model)}

        # Save a running version of checkpoint with a different name
        torch.save(current_chk, self.temp_chk_file)

        if score > self.checkpoint['score']:
            torch.save(current_chk, self.checkpoint_file)
            print('Score improved: ',
                  str(self.checkpoint['score']) + ' to ' + str(score) + ' BEST CHECKPOINT SAVED')
            self.checkpoint = current_chk
        else:
            print('Score did not improve:' + str(score) + ' BEST: ' + str(self.checkpoint['score']))

    @staticmethod
    def get_logger(log_file=None, header=''):

        if os.path.isfile(log_file):
            print('### CRITICAL!!! ' + log_file + '" already exists. OVERRIDE [Y/N]?')
            ui = input()
            if ui == 'N' or ui == 'n':
                sys.exit(1)

        file = open(log_file, 'w')
        NNTrainer.flush(file, header)
        return file

    @staticmethod
    def flush(logger, msg):
        if logger is not None:
            logger.write(msg + '\n')
            logger.flush()
        pass

    @staticmethod
    def adjust_learning_rate(optimizer, epoch):
        if epoch % 40 == 0:
            for param_group in optimizer.param_groups:
                if param_group['lr'] >= 1e-5:
                    param_group['lr'] = param_group['lr'] * 0.5
