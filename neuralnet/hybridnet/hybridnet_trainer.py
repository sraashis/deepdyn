import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from neuralnet.torchtrainer import NNTrainer


class SimpleNNTrainer(NNTrainer):
    def __init__(self, model=None, checkpoint_dir=None, checkpoint_file=None, to_tensorboard=False):
        NNTrainer.__init__(self, model=model, checkpoint_dir=checkpoint_dir, checkpoint_file=checkpoint_file,
                           to_tensorboard=to_tensorboard)

    def train(self, optimizer=None, dataloader=None, epochs=None, use_gpu=False, log_frequency=200,
              validationloader=None):

        if validationloader is None:
            raise ValueError('Please provide validation loader.')

        self.model.train()
        self.model.cuda() if use_gpu else self.model.cpu()
        print('Training...')
        for epoch in range(0, epochs):
            running_loss = 0.0
            accumulated_labels = []
            accumulated_predictions = []
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = Variable(inputs.cuda() if use_gpu else inputs.cpu())
                labels = Variable(labels.cuda() if use_gpu else labels.cpu())

                optimizer.zero_grad()
                outputs = self.model(inputs)

                # print(outputs.shape, labels.shape)
                loss = F.cross_entropy(outputs, labels)
                loss.cuda() if use_gpu else loss.cpu()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                current_loss = loss.item()

                _, predicted = torch.max(outputs, 1)
                # print("##")
                # print(outputs.shape, predicted.shape)
                p, r, f1, s = self.get_score(labels.numpy().squeeze().ravel(), predicted.numpy().squeeze().ravel())

                # Accumulate to calculate score of log frequency batches for better logging
                accumulated_predictions += predicted.numpy().tolist()
                accumulated_labels += labels.numpy().tolist()

                ################## Tensorboard logger setup ###############
                ###########################################################
                if self.to_tenserboard:
                    step = next(self.res['train_counter'])
                    self.logger.scalar_summary('loss/training', current_loss, step)
                    self.logger.scalar_summary('F1/training', f1, step)
                    self.logger.scalar_summary('precision-recall/training', r, p)
                    self.logger.scalar_summary('Support/training', s, step)

                    for tag, value in self.model.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, value.data.cpu().numpy(), step)
                        self.logger.histo_summary(tag + '/gradients', value.grad.data.cpu().numpy(), step)

                    images_to_tb = inputs.view(-1, dataloader.dataset.img_width, dataloader.dataset.img_height)[
                                   :10].data.cpu().numpy()
                    self.logger.image_summary('images/training', images_to_tb, step)
                ###### Tensorboard logger END ##############################
                ############################################################

                if (i + 1) % log_frequency == 0:  # Inspect the loss of every log_frequency batches
                    current_loss = running_loss / log_frequency if (i + 1) % log_frequency == 0 \
                        else (i + 1) % log_frequency
                    p, r, f1, s = self.get_score(np.array(accumulated_labels).ravel(),
                                                 np.array(accumulated_predictions).ravel())
                    running_loss = 0.0
                    accumulated_labels = []
                    accumulated_predictions = []

                print('Epochs:[%d/%d] Batches:[%d/%d], loss:%.3f, pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (epoch + 1, epochs, i + 1, dataloader.__len__(), current_loss, p, r, f1, s),
                      end='\r' if running_loss > 0 else '\n')

            self.checkpoint['epochs'] += 1
            self.evaluate(dataloader=validationloader, use_gpu=use_gpu, force_checkpoint=False, save_best=True)

    def evaluate(self, dataloader=None, use_gpu=False, force_checkpoint=False, save_best=False):

        self.model.eval()
        self.model.cuda() if use_gpu else self.model.cpu()

        all_predictions = []
        all_scores = []
        all_labels = []
        all_IDs = []
        all_patchIJs = []
        print('\nEvaluating...')

        ##### Segment Mode only to use while testing####
        segment_mode = dataloader.dataset.segment_mode
        for i, data in enumerate(dataloader, 0):
            if segment_mode:
                IDs, IJs, inputs, labels = data
            else:
                inputs, labels = data
            inputs = inputs.cuda() if use_gpu else inputs.cpu()
            labels = labels.cuda() if use_gpu else labels.cpu()

            outputs = self.model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)

            # Save scores
            all_predictions += predicted.numpy().tolist()
            all_scores += outputs.data.numpy().tolist()
            all_labels += labels.numpy().tolist()

            ###### For segment mode only ##########
            if segment_mode:
                all_IDs += IDs.numpy().tolist()
                all_patchIJs += IJs.numpy().tolist()
            ##### Segment mode End ###############

            p, r, f1, s = self.get_score(labels.numpy().squeeze().ravel(), predicted.numpy().squeeze().ravel())
            print('Batch[%d/%d] pre:%.3f rec:%.3f f1:%.3f acc:%.3f' % (
                i + 1, dataloader.__len__(), p, r, f1, s),
                  end='\r')

            ########## Feeding to tensorboard starts here...#####################
            ####################################################################
            if self.to_tenserboard:
                step = next(self.res['val_counter'])
                self.logger.scalar_summary('F1/validation', f1, step)
                self.logger.scalar_summary('precision-recall/validation', r, p)
                self.logger.scalar_summary('Acc/validation', s, step)
            #### Tensorfeed stops here# #########################################
            #####################################################################

        print()
        all_IDs = np.array(all_IDs, dtype=np.int)
        all_patchIJs = np.array(all_patchIJs, dtype=np.int)
        all_scores = np.array(all_scores)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        p, r, f1, s = self.get_score(all_labels.ravel(), all_predictions.ravel())
        print('FINAL::: #Precision:%.3f #Recall:%.3f #F1:%.3f #Acc:%.3f' % (p, r, f1, s))
        self._save_if_better(save_best=save_best, force_checkpoint=force_checkpoint, score=f1)

        if segment_mode:
            return all_IDs, all_patchIJs, all_scores, all_predictions, all_labels
        return all_predictions, all_labels
