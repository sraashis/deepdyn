import os
from itertools import count
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from neuralnet.utils.tensorboard_logger import Logger


class NNTrainer:
    def __init__(self, model=None, checkpoint_dir=None, checkpoint_file=None, to_tensorboard=True):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file
        self.checkpoint = NNTrainer._empty_checkpoint()
        self.to_tenserboard = to_tensorboard
        self.logger = Logger(log_dir="./logs/{}".format(time()))
        self.res = {'val_counter': count(), 'train_counter': count()}

    def train(self, optimizer=None, dataloader=None, epochs=None, use_gpu=False, log_frequency=200,
              validationloader=None):

        if validationloader is None:
            raise ValueError('Please provide validation loader.')

        self.model.train()
        self.model.cuda() if use_gpu else self.model.cpu()
        for epoch in range(self.checkpoint['epochs'], self.checkpoint['epochs'] + epochs):
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

                running_loss += loss.data[0]
                current_loss = loss.data[0]

                _, argmax = torch.max(outputs, 1)
                accuracy = (labels == argmax.squeeze()).float().mean()
                if (i + 1) % log_frequency == 0:  # Inspect the loss of every log_frequency batches
                    current_loss = running_loss / log_frequency if (i + 1) % log_frequency == 0 \
                        else (i + 1) % log_frequency
                    running_loss = 0.0

                print('Epoch:[%d/%d] Batches:[%d/%d]   Loss: %.3f Accuracy: %.3f' %
                      (epoch + 1, epochs, i + 1, dataloader.__len__(), current_loss, accuracy),
                      end='\r' if running_loss > 0 else '\n')

                ################## Tensorboard logger setup ###############
                ###########################################################
                if self.to_tenserboard:
                    step = next(self.res['train_counter'])
                    self.logger.scalar_summary('loss/training', current_loss, step)
                    self.logger.scalar_summary('accuracy/training', accuracy.data[0], step)

                    for tag, value in self.model.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, value.data.cpu().numpy(), step)
                        self.logger.histo_summary(tag, value.grad.data.cpu().numpy(), step)

                images_to_tb = inputs.view(-1, dataloader.dataset.patch_size, dataloader.dataset.patch_size)[
                               :12].data.cpu().numpy()
                self.logger.image_summary('images/training', images_to_tb, step)
                ###### Tensorboard logger END ##############################
                ############################################################

            print('\nRunning validation...')
            self.test(dataloader=validationloader, use_gpu=use_gpu, force_checkpoint=False, save_best=True)
        self.checkpoint['epochs'] = self.checkpoint['epochs'] + epochs

    def test(self, dataloader=None, use_gpu=False, force_checkpoint=False, save_best=False):

        self.model.eval()
        self.model.cuda() if use_gpu else self.model.cpu()

        correct = 0
        total = 0
        accuracy = 0.0
        all_predictions = np.array([])
        all_labels = np.array([])
        for i, (inputs, labels) in enumerate(dataloader, 0):
            inputs = inputs.cuda() if use_gpu else inputs.cpu()
            labels = labels.cuda() if use_gpu else labels.cpu()

            outputs = self.model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)

            # Save scores
            all_predictions = np.append(all_predictions, predicted.numpy())
            all_labels = np.append(all_labels, labels.numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print('_________ACCURACY___of___[%d/%d]batches: %.2f%%' % (i + 1, dataloader.__len__(), accuracy), end='\r')

            ########## Feeding to tensorboard starts here...#####################
            ####################################################################
            if self.to_tenserboard:
                step = next(self.res['val_counter'])
                self.logger.scalar_summary('accuracy/validation', accuracy, step)
            #### Tensorfeed stops here# #########################################
            #####################################################################

        if not save_best:
            return int(accuracy), all_predictions, all_labels

        print()
        accuracy = round(accuracy, 3)
        if force_checkpoint:
            self._save_checkpoint(
                NNTrainer._checkpoint(epochs=self.checkpoint['epochs'], model=self.model, accuracy=accuracy))
            print('FORCED checkpoint saved. ')

        last_checkpoint = self._get_last_checkpoint()
        if accuracy > last_checkpoint['accuracy']:
            self._save_checkpoint(
                NNTrainer._checkpoint(epochs=self.checkpoint['epochs'], model=self.model, accuracy=accuracy))
            print('Accuracy improved. __was ' + str(last_checkpoint['accuracy']) + ' [ <CHECKPOINT> saved. ]')

        else:
            last_checkpoint['epochs'] = self.checkpoint['epochs']
            self._save_checkpoint(last_checkpoint)
            print('Accuracy did not improve. __was ' + str(last_checkpoint['accuracy']))

        return int(accuracy), all_predictions, all_labels

    def _save_checkpoint(self, checkpoint):
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, self.checkpoint_file))

    @staticmethod
    def _checkpoint(epochs=None, model=None, accuracy=None):
        return {'state': model.state_dict(),
                'epochs': epochs,
                'accuracy': accuracy,
                'model': str(model)}

    def resume_from_checkpoint(self):
        try:
            self.checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.checkpoint_file))
            self.model.load_state_dict(self.checkpoint['state'])
            print('Resumed from last checkpoint: ' + self.checkpoint_file)
            print(self.checkpoint['model'])
        except Exception as e:
            print('ERROR: ' + str(e))

    def _get_last_checkpoint(self):
        try:
            return torch.load(os.path.join(self.checkpoint_dir, self.checkpoint_file))
        except Exception as e:
            return NNTrainer._empty_checkpoint()

    @staticmethod
    def _empty_checkpoint():
        return {'epochs': 0, 'state': None, 'accuracy': 0.0, 'model': 'EMPTY'}
