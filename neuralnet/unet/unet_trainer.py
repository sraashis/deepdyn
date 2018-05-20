import torch
import torch.nn.functional as F
from torch.autograd import Variable

from neuralnet.torchtrainer import NNTrainer


class UnetNNTrainer(NNTrainer):
    def train(self, optimizer=None, dataloader=None, epochs=None, use_gpu=False, log_frequency=200,
              validationloader=None):

        if validationloader is None:
            raise ValueError('Please provide validation loader.')

        self.model.train()
        self.model.cuda() if use_gpu else self.model.cpu()
        print('Training...')
        for epoch in range(0, epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = Variable(inputs.cuda() if use_gpu else inputs.cpu())
                labels = Variable(labels.cuda() if use_gpu else labels.cpu())

                optimizer.zero_grad()
                outputs = self.model(inputs)
                print('****************')
                print(outputs.shape)
                print(outputs[0])
                loss = F.binary_cross_entropy(outputs, labels)
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

                print('Epochs:[%d/%d] Batches:[%d/%d]   Loss: %.3f accuracy: %.3f' %
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
                        self.logger.histo_summary(tag + '/gradients', value.grad.data.cpu().numpy(), step)

                    images_to_tb = inputs.view(-1, dataloader.dataset.patch_size, dataloader.dataset.patch_size)[
                                   :10].data.cpu().numpy()
                    self.logger.image_summary('images/training', images_to_tb, step)
                ###### Tensorboard logger END ##############################
                ############################################################
            self.checkpoint['epochs'] += 1
            self.evaluate(dataloader=validationloader, use_gpu=use_gpu, force_checkpoint=False, save_best=True)
