import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable


class NNTrainer:
    def __init__(self, model=None, trainloader=None, testloader=None, checkpoint_dir=None, checkpoint_file=None):
        self.best = None
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file
        self.checkpoint = NNTrainer._empty_checkpoint()

    def train(self, optimizer=None, epochs=None, use_gpu=False, override_checkpoint=False):
        print('Starting training...')
        if use_gpu:
            self.model.cuda()

        log_frequency = max(int(4000 / self.trainloader.batch_size), 1)
        self.model.train()
        for epoch in range(self.checkpoint['epochs'], self.checkpoint['epochs'] + epochs):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                if use_gpu:
                    loss.cuda()

                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]
                if i % log_frequency == log_frequency - 1:
                    print('[epoch: %d, batches: %d] loss: %.3f' % (epoch + 1, i + 1, running_loss / log_frequency))
                    running_loss = 0.0
        print('Done with training.')

        accuracy = self.test(use_gpu)
        if override_checkpoint:
            self._checkpoint(epochs, accuracy)
            print('Checkpoint OVERRIDDEN.')

        last_checkpoint = self._last_checkpoint()
        if accuracy > last_checkpoint['accuracy']:
            self._checkpoint(last_checkpoint['epochs'] + epochs, accuracy)
            print('Checkpoint saved:' + self.checkpoint_file)

    def test(self, use_gpu=False):
        if use_gpu:
            self.model.cuda()
        self.model.eval()
        correct = 0
        total = 0
        log_frequency = max(int(2000 / self.testloader.batch_size), 1)
        for i, data in enumerate(self.testloader, 0):
            images, labels = data

            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()

            outputs = self.model(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            acc = 100 * correct / total
            if i % log_frequency == log_frequency - 1:
                print('Accuracy of %d batches: %d %%' % (i + 1, acc))
        return int(acc)

    def _checkpoint(self, epochs=None, accuracy=None):
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

    def _last_checkpoint(self):
        try:
            return torch.load(os.path.join(self.checkpoint_dir, self.checkpoint_file))
        except Exception as e:
            return NNTrainer._empty_checkpoint()

    @staticmethod
    def _empty_checkpoint():
        return {'epochs': 0, 'state': None, 'accuracy': 0.0}
