import numpy as np
import torch
from torch.autograd import Variable

from neuralnet.torchtrainer import NNTrainer


class UnetNNTrainer(NNTrainer):
    def __init__(self, model=None, checkpoint_dir=None, checkpoint_file=None, to_tensorboard=False):
        NNTrainer.__init__(self, model=model, checkpoint_dir=checkpoint_dir, checkpoint_file=checkpoint_file,
                           to_tensorboard=to_tensorboard)

    def evaluate(self, dataloader=None, use_gpu=False, force_checkpoint=False, save_best=False):

        self.model.eval()
        self.model.cuda() if use_gpu else self.model.cpu()

        all_predictions = []
        all_labels = []
        all_scores = []
        print('\nEvaluating...')

        ##### Segment Mode only to use while testing####
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.cuda() if use_gpu else inputs.cpu()
            labels = labels.cuda() if use_gpu else labels.cpu()

            outputs = self.model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            all_scores += outputs.data.numpy().tolist()

            # Save scores
            all_predictions += predicted.numpy().tolist()
            all_labels += labels.numpy().tolist()

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
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)

        p, r, f1, s = self.get_score(all_labels.ravel(), all_predictions.ravel())
        print('FINAL::: #Precision:%.3f #Recall:%.3f #F1:%.3f #Acc:%.3f' % (p, r, f1, s))
        self._save_if_better(save_best=save_best, force_checkpoint=force_checkpoint, score=f1)

        return all_scores, all_predictions, all_labels
