import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class NNTrainer:
    def __init__(self, model=None, checkpoint_dir=None, checkpoint_file=None):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file
        self.checkpoint = NNTrainer._empty_checkpoint()

    def train(self, optimizer=None, dataloader=None, epochs=None, use_gpu=False):

        self.model.train()
        self.model.cuda() if use_gpu else self.model.cpu()

        log_frequency = max(int(4000 / dataloader.batch_size), 1)

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
                if i % log_frequency == log_frequency - 1:
                    print('[epoch: %d, batches: %d] loss: %.3f' % (epoch + 1, i + 1, running_loss / log_frequency))
                    running_loss = 0.0
        self.checkpoint['epochs'] = epochs
        print('Done with training.')

    def test(self, dataloader=None, use_gpu=False, force_checkpoint=False):

        self.model.eval()
        self.model.cuda() if use_gpu else self.model.cpu()

        correct = 0
        total = 0
        accuracy = 0.0
        all_predictions = np.array([])
        all_labels = np.array([])
        log_frequency = max(int(2000 / dataloader.batch_size), 1)
        for i, data in enumerate(dataloader, 0):

            inputs, labels = data
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
            if i % log_frequency == log_frequency - 1:
                print('Accuracy of %d batches: %d %%' % (i + 1, accuracy))

        if force_checkpoint:
            self._save_checkpoint(self.checkpoint['epochs'], accuracy)
            print('Forced checkpoint.')

        last_checkpoint = self._get_last_checkpoint()
        if accuracy > last_checkpoint['accuracy']:
            self._save_checkpoint(last_checkpoint['epochs'], accuracy)
            print('Checkpoint saved:' + self.checkpoint_file)
        return int(accuracy), all_predictions, all_labels

    def _save_checkpoint(self, epochs=None, accuracy=None):
        self.checkpoint = {'state': self.model.state_dict(),
                           'epochs': epochs,
                           'accuracy': accuracy}
        torch.save(self.checkpoint, os.path.join(self.checkpoint_dir, self.checkpoint_file))

    def resume_from_checkpoint(self):
        try:
            self.checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.checkpoint_file))
            self.model.load_state_dict(self.checkpoint['state'])
            print('Resumed: ' + self.checkpoint_file)
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
