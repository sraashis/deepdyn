"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os
import threading

import PIL.Image as IMG
import numpy as np
import torch
import torch.nn.functional as F

import neuralnet.utils.nviz as plt
from neuralnet.utils.measurements import ScoreAccumulator
import sys
import neuralnet.utils.loss as l
import random as rd


class NNTrainer:
    # Possible headers that may appear in log files. These are the ones which are plotted
    _maybe_headers = ['LOSS', 'F1', 'MSE', 'ACCURACY', 'F1', 'DICE']

    def __init__(self, conf=None, model=None, optimizer=None):

        self.conf = conf
        self.log_dir = self.conf.get('Dirs').get('logs', 'net_logs')
        self.epochs = self.conf.get('Params').get('epochs', 100)
        self.log_frequency = self.conf.get('Params').get('log_frequency', 10)
        self.validation_frequency = self.conf.get('Params').get('validation_frequency', 1)
        self.mode = self.conf.get('Params').get('mode', 'test')

        # Initialize necessary logging conf
        self.checkpoint_file = os.path.join(self.log_dir, self.conf.get('checkpoint_file'))
        _log_key = self.conf.get('checkpoint_file').split('.')[0]
        self.test_logger = NNTrainer.get_logger(log_file=os.path.join(self.log_dir, _log_key + '-TEST.csv'),
                                                header=self.log_header.get('test', ''))
        if self.mode == 'train':
            self.train_logger = NNTrainer.get_logger(log_file=os.path.join(self.log_dir, _log_key + '-TRAIN.csv'),
                                                     header=self.log_header.get('train', ''))
            self.val_logger = NNTrainer.get_logger(log_file=os.path.join(self.log_dir, _log_key + '-VAL.csv'),
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

    @property
    def _active_headers(self):
        train = [e for e in self._maybe_headers if e in self.log_header['train'].split(',')]
        val = [e for e in self._maybe_headers if e in self.log_header['validation'].split(',')]
        test = [e for e in self._maybe_headers if e in self.log_header['test'].split(',')]
        return train, val, test

    def test(self, data_loaders=None, gen_images=True):
        self.evaluate(data_loaders, gen_images)
        _, _, h_test = self.active_headers
        for y in h_test:
            plt.y_scatter(file=self.test_log_file, y=y, label='ID', save=True, title='Test')
        if 'PRECISION' in h_test and 'RECALL' in h_test:
            plt.xy_scatter(file=self.test_log_file, save=True, x='PRECISION', y='RECALL', label='ID', title='Test')

    def train(self, data_loader=None, validation_loader=None):
        print('Training...')
        for epoch in range(1, self.epochs + 1):
            self._adjust_learning_rate(epoch=epoch)
            self.checkpoint['total_epochs'] = epoch
            self._run_epoch(epoch=epoch, data_laoder=data_loader)
            self._gen_plots(data_loader=data_loader, validation_loader=validation_loader)
            if epoch % self.validation_frequency == 0:
                self.evaluate(data_loaders=validation_loader, logger=self.val_logger, gen_images=False)
                if self.early_stop(patience=75):
                    return

        if not self.train_logger and not self.train_logger.closed:
            self.train_logger.close()
        if not self.val_logger and not self.val_logger.closed:
            self.val_logger.close()
        if not self.test_logger and not self.test_logger.closed:
            self.test_logger.close()

    def _gen_plots(self, **kw):
        h_train, h_val, h_test = self.active_headers
        self.plot_column_keys(file=self.train_log_file, batches_per_epoch=kw['data_loader'].__len__(),
                              keys=h_train)
        self.plot_column_keys(file=self.val_log_file, batches_per_epoch=kw['validation_loader'].__len__(),
                              keys=h_val)

        if 'PRECISION' in h_train and 'RECALL' in h_train:
            plt.plot_cmap(file=self.train_log_file, save=True, x='PRECISION', y='RECALL')

        if 'PRECISION' in h_val and 'RECALL' in h_val:
            plt.plot_cmap(file=self.val_log_file, save=True, x='PRECISION', y='RECALL')

    def _run_epoch(self, **kw):
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

    @property
    def log_header(self):
        return {
            'train': 'ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS',
            'validation': 'ID,PRECISION,RECALL,F1,ACCURACY',
            'test': 'ID,PRECISION,RECALL,F1,ACCURACY'
        }

    def _train_pixel_wise_mse(self, optimizer=None, data_loader=None, validation_loader=None):

        if validation_loader is None:
            raise ValueError('Please provide validation loader.')

        logger = NNTrainer.get_logger(self.train_log_file,
                                      header='ID,EPOCH,BATCH,LOSS')

        val_logger = NNTrainer.get_logger(self.validation_log_file,
                                          header='ID,LOSS')

        print('Training...')
        for epoch in range(1, self.epochs + 1):

            self.model.train()
            running_loss = 0.0
            self._adjust_learning_rate(optimizer=optimizer, epoch=epoch)
            self.checkpoint['total_epochs'] = epoch

            for i, data in enumerate(data_loader, 1):
                inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).float()

                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = F.mse_loss(outputs.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                running_loss += current_loss

                if i % self.log_frequency == 0:
                    print('Epochs[%d/%d] Batch[%d/%d] MSE loss:%.5f ' %
                          (
                              epoch, self.epochs, i, data_loader.__len__(), running_loss / self.log_frequency))
                    running_loss = 0.0

                self.flush(logger, ','.join(str(x) for x in [0, epoch, i, current_loss]))

            self.plot_train(file=self.train_log_file, batches_per_epochs=data_loader.__len__(), keys=['LOSS'])
            if epoch % self.validation_frequency == 0:
                self.evaluate(data_loaders=validation_loader, logger=val_logger, gen_images=False)
                if self.early_stop(patience=75):
                    return

            self.plot_val(self.validation_log_file, batches_per_epoch=len(validation_loader))

        try:
            logger.close()
            val_logger.close()
        except IOError:
            pass

    def _train_pixel_wise_dice_loss(self, optimizer=None, data_loader=None, validation_loader=None):

        if validation_loader is None:
            raise ValueError('Please provide validation loader.')

        logger = NNTrainer.get_logger(self.train_log_file,
                                      header='ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS')

        val_logger = NNTrainer.get_logger(self.validation_log_file,
                                          header='ID,PRECISION,RECALL,F1,ACCURACY')

        print('Training...')
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            score_acc = ScoreAccumulator()
            running_loss = 0.0
            self._adjust_learning_rate(optimizer=optimizer, epoch=epoch)
            self.checkpoint['total_epochs'] = epoch
            for i, data in enumerate(data_loader, 1):
                inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).long()

                optimizer.zero_grad()
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                # This was used for our experiment
                # loss = l.dice_loss(outputs[:, 1, :, :], labels, beta=rd.choice(np.arange(1, 2, 0.1).tolist()))
                loss = l.dice_loss(outputs[:, 1, :, :], labels, beta=1)
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                running_loss += current_loss
                p, r, f1, a = score_acc.reset().add_tensor(predicted, labels).get_prfa()
                if i % self.log_frequency == 0:
                    print('Epochs[%d/%d] Batch[%d/%d] loss:%.5f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                          (
                              epoch, self.epochs, i, data_loader.__len__(), running_loss / self.log_frequency, p, r, f1,
                              a))
                    running_loss = 0.0

                self.flush(logger, ','.join(str(x) for x in [0, epoch, i, p, r, f1, a, current_loss]))

            self.plot_train(file=self.train_log_file, batches_per_epochs=data_loader.__len__(), keys=['LOSS', 'F1'])
            if epoch % self.validation_frequency == 0:
                self.evaluate(data_loaders=validation_loader, logger=val_logger, gen_images=False)
                if self.early_stop(patience=20):
                    return

            self.plot_val(self.validation_log_file, batches_per_epoch=len(validation_loader))

        try:
            logger.close()
            val_logger.close()
        except IOError:
            pass

    def evaluate(self, data_loaders=None, gen_images=False):
        return NotImplementedError('------Evaluation can vary a lot.. Needs to be implemented.-------')

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
        NNTrainer.flush(file, header)
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

    # @staticmethod
    # def plot_train(file=None, keys=None, batches_per_epochs=None):
    #
    #     def f(fl=file, ks=keys, bpep=batches_per_epochs):
    #         plt.plot_cmap(file=fl, save=True, x='PRECISION', y='RECALL', title='Training')
    #         for k in ks:
    #             plt.plot(file=fl, y=k, title='Training', x_tick_skip=bpep, save=True)
    #
    #     NNTrainer.send_to_back(func=f)

    @staticmethod
    def plot_pr_color_map(file, title=''):
        def f(fl=file, tl=title):
            plt.plot_cmap(file=fl, save=True, x='PRECISION', y='RECALL', title=tl)

        NNTrainer.send_to_back(f)

    @staticmethod
    def plot_column_keys(file, batches_per_epoch, title='', keys=[]):
        # LOSS, MSE, F1, ACCURACY
        def f(fl=file, b_per_ep=batches_per_epoch):
            for k in keys:
                plt.plot(file=fl, title=title, y=k, save=True,
                         x_tick_skip=b_per_ep)

        NNTrainer.send_to_back(f)

    # @staticmethod
    # def plot_test(file):
    #     def f(fl=file):
    #         plt.y_scatter(file=fl, y='F1', label='ID', save=True, title='Test')
    #         plt.y_scatter(file=fl, y='ACCURACY', label='ID', save=True, title='Test')
    #         plt.xy_scatter(file=fl, save=True, x='PRECISION', y='RECALL', label='ID', title='Test')
    #
    #     NNTrainer.send_to_back(f)

    @staticmethod
    def send_to_back(func, kwargs={}):
        t = threading.Thread(target=func, kwargs=kwargs)
        t.start()
