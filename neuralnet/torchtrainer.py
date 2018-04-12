import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler


class NNTrainer:
    def __init__(self, model=None, checkpoint_dir=None, checkpoint_file=None):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file
        self.checkpoint = NNTrainer._empty_checkpoint()

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
                if (i + 1) % log_frequency == 0:  # Inspect the loss of every log_frequency batches
                    current_loss = running_loss / log_frequency
                    running_loss = 0.0

                print('epoch:[%d/%d] batches:[%d/%d]   Loss = %.3f' %
                      (epoch + 1, epochs, i + 1, dataloader.__len__(), current_loss),
                      end='\r' if running_loss > 0 else '\n')
            print()
            self.test(dataloader=validationloader, use_gpu=use_gpu, force_checkpoint=False)
        self.checkpoint['epochs'] = self.checkpoint['epochs'] + epochs

    def test(self, dataloader=None, use_gpu=False, force_checkpoint=False):

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
            print('_________ACCURACY___of___[%d/%d]_batches: %.2f%%' % (i + 1, dataloader.__len__(), accuracy), end='\r')
        print()
        if force_checkpoint:
            self._save_checkpoint(
                NNTrainer._checkpoint(epochs=self.checkpoint['epochs'], model=self.model, accuracy=accuracy))
            print('FORCED checkpoint saved. ')

        last_checkpoint = self._get_last_checkpoint()
        if accuracy > last_checkpoint['accuracy']:
            self._save_checkpoint(
                NNTrainer._checkpoint(epochs=self.checkpoint['epochs'], model=self.model, accuracy=accuracy))
            print('Accuracy improved which was ' + str(last_checkpoint['accuracy']) + ' [ <CHECKPOINT> saved. ]')

        else:
            last_checkpoint['epochs'] = self.checkpoint['epochs']
            self._save_checkpoint(last_checkpoint)
            print('Accuracy did not improve which was ' + str(last_checkpoint['accuracy']))

        return int(accuracy), all_predictions, all_labels

    def _save_checkpoint(self, checkpoint):
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, self.checkpoint_file))

    @staticmethod
    def _checkpoint(epochs=None, model=None, accuracy=None):
        return {'state': model.state_dict(),
                'epochs': epochs,
                'accuracy': accuracy}

    def resume_from_checkpoint(self):
        try:
            self.checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.checkpoint_file))
            self.model.load_state_dict(self.checkpoint['state'])
            print('Resumed from last checkpoint: ' + self.checkpoint_file)
        except Exception as e:
            print('ERROR: ' + str(e))

    def _get_last_checkpoint(self):
        try:
            return torch.load(os.path.join(self.checkpoint_dir, self.checkpoint_file))
        except Exception as e:
            return NNTrainer._empty_checkpoint()

    @staticmethod
    def _empty_checkpoint():
        return {'epochs': 0, 'state': None, 'accuracy': 0.0}
