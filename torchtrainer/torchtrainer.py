"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os
import random as rd
import sys

import numpy as np
import torch
import torch.nn.functional as F

from utils.loss import dice_loss
from utils.measurements import ScoreAccumulator


class NNTrainer:

    def __init__(self, conf=None, model=None, optimizer=None):

        # Initialize parameters and directories before-hand so that we can clearly track which ones are used
        self.conf = conf
        self.log_dir = self.conf.get('Dirs').get('logs', 'net_logs')
        self.epochs = self.conf.get('Params').get('epochs', 100)
        self.log_frequency = self.conf.get('Params').get('log_frequency', 10)
        self.validation_frequency = self.conf.get('Params').get('validation_frequency', 1)
        self.mode = self.conf.get('Params').get('mode', 'test')

        # Initialize necessary logging conf
        self.checkpoint_file = os.path.join(self.log_dir, self.conf.get('checkpoint_file'))

        self.log_headers = self.get_log_headers()
        _log_key = self.conf.get('checkpoint_file').split('.')[0]
        self.test_logger = NNTrainer.get_logger(log_file=os.path.join(self.log_dir, _log_key + '-TEST.csv'),
                                                header=self.log_headers.get('test', ''))
        if self.mode == 'train':
            self.train_logger = NNTrainer.get_logger(log_file=os.path.join(self.log_dir, _log_key + '-TRAIN.csv'),
                                                     header=self.log_headers.get('train', ''))
            self.val_logger = NNTrainer.get_logger(log_file=os.path.join(self.log_dir, _log_key + '-VAL.csv'),
                                                   header=self.log_headers.get('validation', ''))

        #  Function to initialize class weights, default is [1, 1]
        self.dparm = self.conf.get("Funcs").get('dparm')

        # Handle gpu/cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda" if self.conf['Params'].get('use_gpu', False) else "cpu")
        else:
            print('### GPU not found.')
            self.device = torch.device("cpu")

        # Initialization to save model
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.checkpoint = {'total_epochs:': 0, 'epochs': 0, 'state': None, 'score': 0.0, 'model': 'EMPTY'}
        self.patience = self.conf.get('Params').get('patience', 40)

    def test(self, data_loaders=None):
        print('Running test')
        score = ScoreAccumulator()
        self.model.eval()
        with torch.no_grad():
            self.evaluate(data_loaders=data_loaders, gen_images=True, score_acc=score, logger=self.test_logger)
        self._on_test_end(log_file=self.test_logger.name)
        if not self.test_logger and not self.test_logger.closed:
            self.test_logger.close()

    def evaluate(self, data_loaders=None, logger=None, gen_images=False, score_acc=None):
        return NotImplementedError('------Evaluation step can vary a lot.. Needs to be implemented.-------')

    def _on_test_end(self, **kw):
        pass

    def train(self, data_loader=None, validation_loader=None, epoch_run=None):
        print('Training...')
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self._adjust_learning_rate(epoch=epoch)
            self.checkpoint['total_epochs'] = epoch

            # Run one epoch
            epoch_run(epoch=epoch, data_loader=data_loader, logger=self.train_logger)

            self._on_epoch_end(data_loader=data_loader, log_file=self.train_logger.name)

            # Validation_frequency is the number of epoch until validation
            if epoch % self.validation_frequency == 0:
                print('############# Running validation... ####################')
                self.model.eval()
                with torch.no_grad():
                    self.validation(epoch=epoch, validation_loader=validation_loader, epoch_run=epoch_run)
                self._on_validation_end(data_loader=validation_loader, log_file=self.val_logger.name)
                if self.early_stop(patience=self.patience):
                    return
                print('########################################################')

        if not self.train_logger and not self.train_logger.closed:
            self.train_logger.close()
        if not self.val_logger and not self.val_logger.closed:
            self.val_logger.close()

    def _on_epoch_end(self, **kw):
        pass

    def _on_validation_end(self, **kw):
        pass

    def get_log_headers(self):
        # EXAMPLE:
        # return {
        #     'train': 'ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS',
        #     'validation': 'ID,PRECISION,RECALL,F1,ACCURACY',
        #     'test': 'ID,PRECISION,RECALL,F1,ACCURACY'
        # }
        raise NotImplementedError('Must be implemented to use.')

    def validation(self, epoch=None, validation_loader=None, epoch_run=None):
        score_acc = ScoreAccumulator()
        self.evaluate(data_loaders=validation_loader, logger=self.val_logger, gen_images=False, score_acc=score_acc)
        # epoch_run(epoch=epoch, data_loader=validation_loader, logger=self.val_logger, score_acc=score_acc)
        p, r, f1, a = score_acc.get_prfa()
        print('>>> PRF1: ', [p, r, f1, a])
        self._save_if_better(score=f1)

    def resume_from_checkpoint(self, parallel_trained=False):
        self.checkpoint = torch.load(self.checkpoint_file)
        print(self.checkpoint_file, ' Loaded...')
        try:
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
        except Exception as e:
            print('ERROR: ' + str(e))

    def _save_if_better(self, score=None):

        if self.mode == 'test':
            return

        if score > self.checkpoint['score']:
            print('Score improved: ',
                  str(self.checkpoint['score']) + ' to ' + str(score) + ' BEST CHECKPOINT SAVED')
            self.checkpoint['state'] = self.model.state_dict()
            self.checkpoint['epochs'] = self.checkpoint['total_epochs']
            self.checkpoint['score'] = score
            self.checkpoint['model'] = str(self.model)
            torch.save(self.checkpoint, self.checkpoint_file)
        else:
            print('Score did not improve:' + str(score) + ' BEST: ' + str(self.checkpoint['score']) + ' Best EP: ' + (
                str(self.checkpoint['epochs'])))

    def early_stop(self, patience=35):
        return self.checkpoint['total_epochs'] - self.checkpoint['epochs'] >= patience * self.validation_frequency

    @staticmethod
    def get_logger(log_file=None, header=''):

        if os.path.isfile(log_file):
            print('### CRITICAL!!! ' + log_file + '" already exists.')
            ip = input('Override? [Y/N]: ')
            if ip == 'N' or ip == 'n':
                sys.exit(1)

        file = open(log_file, 'w')
        NNTrainer.flush(file, header)
        return file

    @staticmethod
    def flush(logger, msg):
        if logger is not None:
            logger.write(msg + '\n')
            logger.flush()

    def _adjust_learning_rate(self, epoch):
        if epoch % 30 == 0:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] >= 1e-5:
                    param_group['lr'] = param_group['lr'] * 0.7

    @staticmethod
    def plot_column_keys(file, batches_per_epoch, title='', keys=[]):
        """
        This method plots all desired columns, specified in key, from log file
        :param file:
        :param batches_per_epoch:
        :param title:
        :param keys:
        :return:
        """
        from viz.nviz import plot
        for k in keys:
            plot(file=file, title=title, y=k, save=True,
                 x_tick_skip=batches_per_epoch)

    '''
    ######################################################################################
    Below are the functions specific to loss function and training strategy
    These functions should be passed while calling *TorchTrainer.train() from main.py
    ######################################################################################
    '''

    def epoch_ce_loss(self, **kw):
        """
        One epoch implementation of binary cross-entropy loss
        :param kw:
        :return:
        """
        running_loss = 0.0
        score_acc = ScoreAccumulator() if self.model.training else kw.get('score_acc')
        assert isinstance(score_acc, ScoreAccumulator)

        for i, data in enumerate(kw['data_loader'], 1):
            inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).long()

            if self.model.training:
                self.optimizer.zero_grad()

            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)

            loss = F.cross_entropy(outputs, labels,
                                   weight=torch.FloatTensor(self.dparm(self.conf)).to(self.device))

            if self.model.training:
                loss.backward()
                self.optimizer.step()

            current_loss = loss.item()
            running_loss += current_loss

            if self.model.training:
                score_acc.reset()

            p, r, f1, a = score_acc.add_tensor(predicted, labels).get_prfa()

            if i % self.log_frequency == 0:
                print('Epochs[%d/%d] Batch[%d/%d] loss:%.5f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (
                          kw['epoch'], self.epochs, i, kw['data_loader'].__len__(),
                          running_loss / self.log_frequency, p, r, f1,
                          a))
                running_loss = 0.0
            self.flush(kw['logger'],
                       ','.join(str(x) for x in [0, kw['epoch'], i, p, r, f1, a, current_loss]))

    def epoch_dice_loss(self, **kw):

        score_acc = ScoreAccumulator() if self.model.training else kw.get('score_acc')
        assert isinstance(score_acc, ScoreAccumulator)

        running_loss = 0.0
        for i, data in enumerate(kw['data_loader'], 1):
            inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).long()

            if self.model.training:
                self.optimizer.zero_grad()

            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)

            outputs_softmax = F.softmax(outputs, 1)
            loss1 = dice_loss(outputs_softmax[:, 1, :, :], labels, beta=rd.choice(np.arange(1, 2, 0.1).tolist()))
            loss2 = dice_loss(outputs_softmax[:, 0, :, :], 1 - labels,
                              beta=rd.choice(np.arange(1, 2, 0.1).tolist()))
            loss = loss1 + loss2
            if self.model.training:
                loss.backward()
                self.optimizer.step()

            current_loss = loss.item()
            running_loss += current_loss

            if self.model.training:
                score_acc.reset()
            p, r, f1, a = score_acc.add_tensor(predicted, labels).get_prfa()

            if i % self.log_frequency == 0:
                print('Epochs[%d/%d] Batch[%d/%d] loss:%.5f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (
                          kw['epoch'], self.epochs, i, kw['data_loader'].__len__(), running_loss / self.log_frequency,
                          p, r, f1,
                          a))
                running_loss = 0.0

            self.flush(kw['logger'], ','.join(str(x) for x in [0, kw['epoch'], i, p, r, f1, a, current_loss]))

    def epoch_mse_loss(self, **kw):

        # Todo score accumulation to check if this model is better than the saved one
        running_loss = 0.0
        for i, data in enumerate(kw['data_loader'], 1):
            inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).float()

            if self.model.training:
                self.optimizer.zero_grad()

            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)

            if len(labels.shape) == 3:
                labels = torch.unsqueeze(labels, 1)

            loss = F.mse_loss(outputs, labels)

            if self.model.training:
                loss.backward()
                self.optimizer.step()

            current_loss = loss.item()
            running_loss += current_loss

            if i % self.log_frequency == 0:
                print('Epochs[%d/%d] Batch[%d/%d] MSE loss:%.5f ' %
                      (
                          kw['epoch'], self.epochs, i, kw['data_loader'].__len__(), running_loss / self.log_frequency))
                running_loss = 0.0

            self.flush(kw['logger'], ','.join(str(x) for x in [0, kw['epoch'], i, current_loss]))
