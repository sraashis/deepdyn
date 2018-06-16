import numpy as np
import torch
from torch.autograd import Variable

import neuralnet.utils.measurements as mggmt
from neuralnet.torchtrainer import NNTrainer


class HybridNNTrainer(NNTrainer):
    def __init__(self, model=None, checkpoint_dir=None, checkpoint_file=None, log_to_file=False):
        NNTrainer.__init__(self, model=model, checkpoint_dir=checkpoint_dir, checkpoint_file=checkpoint_file,
                           log_to_file=log_to_file)

    def evaluate(self, dataloader=None, use_gpu=False, force_checkpoint=False, save_best=False):

        self.model.eval()
        self.model.cuda() if use_gpu else self.model.cpu()

        print('\nEvaluating...')
        TP, FP, TN, FN = [0] * 4
        all_predictions = []
        all_scores = []
        all_labels = []
        all_patchIJs = []

        ##### Segment Mode only to use while testing####
        mode = dataloader.dataset.mode
        for i, data in enumerate(dataloader, 0):
            if mode == 'eval':
                IDs, IJs, inputs, labels = data
            else:
                inputs, labels = data
            inputs = Variable(inputs.cuda() if use_gpu else inputs.cpu())
            labels = Variable(labels.cuda() if use_gpu else labels.cpu())

            outputs = self.model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)

            # Accumulate scores
            all_scores += outputs.data.clone().cpu().numpy().tolist()
            all_predictions += predicted.data.clone().cpu().numpy().tolist()
            all_labels += labels.data.clone().cpu().numpy().tolist()

            ###### For segment mode only ##########
            if mode == 'eval':
                all_patchIJs += IJs.numpy().tolist()
            ##### Segment mode End ###############

            _tp, _fp, _tn, _fn = self.get_score(labels, predicted)
            TP += _tp
            TN += _tn
            FP += _fp
            FN += _fn
            p, r, f1, a = mggmt.get_prf1a(TP, FP, TN, FN)

            print('Batch[%d/%d] pre:%.3f rec:%.3f f1:%.3f acc:%.3f' % (
                i + 1, dataloader.__len__(), p, r, f1, a),
                  end='\r')

            ########## Feeding to tensorboard starts here...#####################
            ####################################################################
            if self.to_tenserboard:
                step = next(self.res['val_counter'])
                self.logger.scalar_summary('F1/validation', f1, step)
                self.logger.scalar_summary('Acc/validation', a, step)
            #### Tensorfeed stops here# #########################################
            #####################################################################

        print()
        all_patchIJs = np.array(all_patchIJs, dtype=np.int)
        all_scores = np.array(all_scores)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        self._save_if_better(save_best=save_best, force_checkpoint=force_checkpoint, score=f1)

        if mode == 'eval':
            return all_patchIJs, all_scores, all_predictions, all_labels
        return all_predictions, all_labels
