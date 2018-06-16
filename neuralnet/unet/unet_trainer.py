import numpy as np
import torch
from torch.autograd import Variable

import neuralnet.utils.measurements as mggmt
from neuralnet.torchtrainer import NNTrainer


class UNetNNTrainer(NNTrainer):
    def __init__(self, model=None, checkpoint_dir=None, checkpoint_file=None, log_to_file=True):
        NNTrainer.__init__(self, model=model, checkpoint_dir=checkpoint_dir, checkpoint_file=checkpoint_file,
                           log_to_file=log_to_file)

    def evaluate(self, dataloader=None, use_gpu=False, force_checkpoint=False, save_best=False):

        self.model.cuda() if use_gpu else self.model.cpu()
        self.model.eval()
        print('\nEvaluating...')
        TP, FP, TN, FN = [0] * 4
        all_predictions = []
        all_scores = []
        all_labels = []

        ##### Segment Mode only to use while testing####
        mode = dataloader.dataset.mode
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.cuda() if use_gpu else inputs.cpu()
            labels = labels.cuda() if use_gpu else labels.cpu()

            outputs = self.model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)

            # Accumulate scores
            all_scores += outputs.data.clone().cpu().numpy().tolist()
            all_predictions += predicted.clone().cpu().numpy().tolist()
            all_labels += labels.clone().cpu().numpy().tolist()

            _tp, _fp, _tn, _fn = self.get_score(labels, predicted)
            TP += _tp
            TN += _tn
            FP += _fp
            FN += _fn
            p, r, f1, a = mggmt.get_prf1a(TP, FP, TN, FN)

            self._log(','.join(str(x) for x in [1, 0, i + 1, p, r, a, f1]))
            print('Batch[%d/%d] pre:%.3f rec:%.3f f1:%.3f acc:%.3f' % (
                i + 1, dataloader.__len__(), p, r, f1, a),
                  end='\r')

        print()
        all_scores = np.array(all_scores)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        self._save_if_better(save_best=save_best, force_checkpoint=force_checkpoint, score=f1)

        if mode == 'eval':
            return all_scores, all_predictions, all_labels
        return all_predictions, all_labels
