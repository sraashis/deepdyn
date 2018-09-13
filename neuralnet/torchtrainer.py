import os
import random
import sys

import torch
import torch.nn.functional as F

from neuralnet.utils.measurements import ScoreAccumulator


class NNTrainer:

    def __init__(self, run_conf=None, model=None):

        self.run_conf = run_conf
        self.log_dir = self.run_conf.get('Dirs').get('logs')
        self.use_gpu = self.run_conf['Params']['use_gpu']
        self.epochs = self.run_conf.get('Params').get('epochs')
        self.log_frequency = self.run_conf.get('Params').get('log_frequency')
        self.validation_frequency = self.run_conf.get('Params').get('validation_frequency')
        self.force_checkpoint = self.run_conf.get('Params').get('force_checkpoint')

        self.checkpoint_file = os.path.join(self.log_dir, self.run_conf.get('Params').get('checkpoint_file'))
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
                          (epoch, self.epochs, i, data_loader.__len__(), running_loss/self.log_frequency, p, r, f1, a))
                    running_loss = 0.0

                self.flush(logger, ','.join(str(x) for x in [0, 0, epoch, i, p, r, f1, a, current_loss]))

            self.checkpoint['epochs'] += 1
            if epoch % self.validation_frequency == 0:
                self.evaluate(data_loaders=validation_loader, force_checkpoint=self.force_checkpoint, logger=logger,
                              mode='train')
        try:
            logger.close()
        except IOError:
            pass

    def evaluate(self, data_loaders=None, force_checkpoint=False, logger=None, mode=None):
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
    def get_logger(log_file=None, header=''):

        if os.path.isfile(log_file):
            print('### CRITICAL!!! ' + log_file + '" already exists. Rename or delete to proceed.')
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
