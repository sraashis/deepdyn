"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os
import random as rd
import sys
import threading

import numpy as np
import torch
import torch.nn.functional as F

import nnbee.utils.loss as l
import nnbee.viz.nviz as plt
from nnbee.utils.measurements import ScoreAccumulator


class NNBee:
    """
    Possible headers that may appear in log files. These are the ones which are plotted
    """
    _maybe_headers = ['LOSS', 'F1', 'MSE', 'ACCURACY', 'F1', 'DICE']

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
        _log_key = self.conf.get('checkpoint_file').split('.')[0]
        self.test_logger = NNBee.get_logger(log_file=os.path.join(self.log_dir, _log_key + '-TEST.csv'),
                                            header=self.log_header.get('test', ''))
        if self.mode == 'train':
            self.train_logger = NNBee.get_logger(log_file=os.path.join(self.log_dir, _log_key + '-TRAIN.csv'),
                                                 header=self.log_header.get('train', ''))
            self.val_logger = NNBee.get_logger(log_file=os.path.join(self.log_dir, _log_key + '-VAL.csv'),
                                               header=self.log_header.get('validation', ''))

        self.active_headers = self._active_headers

        #  Function to initialize class weights, default is [1, 1]
        self.dparm = self.conf.get("Funcs").get('dparm')
        if not self.dparm:
            self.dparm = lambda x: [1.0, 1.0]

        # Handle gpu/cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda" if self.conf['Params'].get('use_gpu', False) else "cpu")
        else:
            print('### GPU not found.')
            self.device = torch.device("cpu")

        # Initialization to save model
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.model_trace = []
        self.checkpoint = {'total_epochs:': 0, 'epochs': 0, 'state': None, 'score': 0.0, 'model': 'EMPTY'}
        self.patience = self.conf.get('Params').get('patience', 35)

    @property
    def _active_headers(self):
        """
        All the headers that are present in log file are filtered in order to generate graph
        :return: None
        """
        train = [e for e in self._maybe_headers if e in self.log_header['train'].split(',')]
        val = [e for e in self._maybe_headers if e in self.log_header['validation'].split(',')]
        test = [e for e in self._maybe_headers if e in self.log_header['test'].split(',')]
        return train, val, test

    def test(self, data_loaders=None, gen_images=True):
        self.model.eval()
        score = ScoreAccumulator()
        self._eval(data_loaders, gen_images, score)
        _, _, h_test = self.active_headers
        for y in h_test:
            plt.y_scatter(file=self.test_log_file, y=y, label='ID', save=True, title='Test')
        if 'PRECISION' in h_test and 'RECALL' in h_test:
            plt.xy_scatter(file=self.test_log_file, save=True, x='PRECISION', y='RECALL', label='ID', title='Test')

        if not self.test_logger and not self.test_logger.closed:
            self.test_logger.close()

    def train(self, data_loader=None, validation_loader=None, epoch_run=None):
        print('Training...')
        for epoch in range(1, self.epochs + 1):
            self._adjust_learning_rate(epoch=epoch)
            self.checkpoint['total_epochs'] = epoch

            # Run on epoch
            epoch_run(epoch=epoch, data_laoder=data_loader)
            self._gen_plots(data_loader=data_loader, validation_loader=validation_loader)

            # Validation_frequency is the number of epoch until validation
            if epoch % self.validation_frequency == 0:
                self._validation(data_loaders=validation_loader, gen_images=False)
                if self.early_stop(patience=self.patience):
                    return

        if not self.train_logger and not self.train_logger.closed:
            self.train_logger.close()
        if not self.val_logger and not self.val_logger.closed:
            self.val_logger.close()
        if not self.test_logger and not self.test_logger.closed:
            self.test_logger.close()

    def _gen_plots(self, **kw):
        h_train, h_val, h_test = self.active_headers

        if kw.get('data_loader'):
            self.plot_column_keys(file=self.train_log_file, batches_per_epoch=kw['data_loader'].__len__(),
                                  keys=h_train)
            if 'PRECISION' in h_train and 'RECALL' in h_train:
                plt.plot_cmap(file=self.train_log_file, save=True, x='PRECISION', y='RECALL')

        if kw.get('validation_loader'):
            self.plot_column_keys(file=self.val_log_file, batches_per_epoch=kw['validation_loader'].__len__(),
                                  keys=h_val)
            if 'PRECISION' in h_val and 'RECALL' in h_val:
                plt.plot_cmap(file=self.val_log_file, save=True, x='PRECISION', y='RECALL')

    @property
    def log_header(self):
        return {
            'train': 'ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS',
            'validation': 'ID,PRECISION,RECALL,F1,ACCURACY',
            'test': 'ID,PRECISION,RECALL,F1,ACCURACY'
        }

    def _validation(self, data_loaders=None, gen_images=False):
        self.model.eval()
        val_score = ScoreAccumulator()
        self._eval(data_loaders=data_loaders, gen_images=gen_images, score_acc=val_score)
        self._gen_plots(validation_loader=data_loaders)
        self._save_if_better(score=val_score.get_prfa()[2])

    def _eval(self, data_loaders=None, gen_images=False, score_acc=None):
        return NotImplementedError('------Evaluation step can vary a lot.. Needs to be implemented.-------')

    def resume_from_checkpoint(self, parallel_trained=False):
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
            print('Score did not improve:' + str(score) + ' BEST: ' + str(self.checkpoint['score']) + ' EP: ' + (
                str(self.checkpoint['epochs'])))

    def early_stop(self, patience=35):
        return self.checkpoint['total_epochs'] - self.checkpoint['epochs'] >= patience * self.validation_frequency

    @staticmethod
    def get_logger(log_file=None, header=''):

        if os.path.isfile(log_file):
            print('### CRITICAL!!! ' + log_file + '" already exists. PLEASE BACKUP. Exiting..')
            sys.exit(1)

        file = open(log_file, 'w')
        NNBee.flush(file, header)
        return file

    @staticmethod
    def flush(logger, msg):
        if logger is not None:
            logger.write(msg + '\n')
            logger.flush()

    @property
    def train_log_file(self):
        if self.train_logger:
            return os.path.basename(self.train_logger.name)

    @property
    def val_log_file(self):
        if self.val_logger:
            return os.path.basename(self.val_logger.name)

    @property
    def test_log_file(self):
        if self.test_logger:
            return os.path.basename(self.test_logger.name)

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

        def f(fl=file, b_per_ep=batches_per_epoch):
            for k in keys:
                plt.plot(file=fl, title=title, y=k, save=True,
                         x_tick_skip=b_per_ep)

        NNBee.send_to_back(f)

    @staticmethod
    def send_to_back(func, kwargs={}):
        t = threading.Thread(target=func, kwargs=kwargs)
        t.start()

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
        score_acc = ScoreAccumulator()
        for i, data in enumerate(kw['data_loader'], 1):
            inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).long()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)

            loss = F.nll_loss(outputs, labels, weight=torch.FloatTensor(self.dparm(self.conf)).to(self.device))
            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()
            running_loss += current_loss
            p, r, f1, a = score_acc.reset().add_tensor(predicted, labels).get_prfa()

            if kw['iter'] % self.log_frequency == 0:
                print('Epochs[%d/%d] Batch[%d/%d] loss:%.5f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (
                          kw['epoch'], self.epochs, i, kw['data_loader'].__len__(),
                          running_loss / self.log_frequency, p, r, f1,
                          a))
                running_loss = 0.0
            self.flush(self.train_logger,
                       ','.join(str(x) for x in [0, kw['epoch'], i, p, r, f1, a, current_loss]))

    def epoch_dice_loss(self, **kw):
        score_acc = ScoreAccumulator()
        running_loss = 0.0
        for i, data in enumerate(kw['data_loader'], 1):
            inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).long()
            # weights = data['weights'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Balancing imbalanced class as per computed weights from the dataset
            # w = torch.FloatTensor(2).random_(1, 100).to(self.device)
            # wd = torch.FloatTensor(*labels.shape).uniform_(0.1, 2).to(self.device)

            loss = l.dice_loss(outputs[:, 1, :, :], labels, beta=rd.choice(np.arange(1, 2, 0.1).tolist()))
            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()
            running_loss += current_loss
            p, r, f1, a = score_acc.reset().add_tensor(predicted, labels).get_prfa()
            if i % self.log_frequency == 0:
                print('Epochs[%d/%d] Batch[%d/%d] loss:%.5f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (
                          kw['epoch'], self.epochs, i, kw['data_loader'].__len__(), running_loss / self.log_frequency,
                          p, r, f1,
                          a))
                running_loss = 0.0

            self.flush(self.train_logger, ','.join(str(x) for x in [0, kw['epoch'], i, p, r, f1, a, current_loss]))

    def epoch_mse_loss(self, **kw):
        for epoch in range(1, self.epochs + 1):

            running_loss = 0.0
            for i, data in enumerate(kw['data_loader'], 1):
                inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                loss = F.mse_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                current_loss = loss.item()
                running_loss += current_loss

                if i % self.log_frequency == 0:
                    print('Epochs[%d/%d] Batch[%d/%d] MSE loss:%.5f ' %
                          (
                              epoch, self.epochs, i, kw['data_loader'].__len__(), running_loss / self.log_frequency))
                    running_loss = 0.0

                self.flush(self.train_logger, ','.join(str(x) for x in [0, epoch, i, current_loss]))
