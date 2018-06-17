import numpy as np
import torch

import neuralnet.utils.measurements as mggmt
from neuralnet.torchtrainer import NNTrainer


class UNetNNTrainer(NNTrainer):
    def __init__(self, model=None, checkpoint_dir=None, checkpoint_file=None, log_to_file=True, use_gpu=True):
        NNTrainer.__init__(self, model=model, checkpoint_dir=checkpoint_dir, checkpoint_file=checkpoint_file,
                           log_to_file=log_to_file, use_gpu=use_gpu)

    def _evaluate(self, dataloader=None, force_checkpoint=False, save_best=False):
        TP, FP, TN, FN = [0] * 4
        all_predictions = []
        all_scores = []
        all_labels = []

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Accumulate scores
            all_scores += outputs.clone().cpu().numpy().tolist()
            all_predictions += predicted.clone().cpu().numpy().tolist()
            all_labels += labels.clone().cpu().numpy().tolist()
            _tp, _fp, _tn, _fn = self.get_score(labels, predicted)
            TP += _tp
            TN += _tn
            FP += _fp
            FN += _fn
            p, r, f1, a = mggmt.get_prf1a(TP, FP, TN, FN)
            self._log(','.join(str(x) for x in [1, 0, i + 1, p, r, f1, a]))
            print('Batch[%d/%d] pre:%.3f rec:%.3f f1:%.3f acc:%.3f' % (
                i + 1, dataloader.__len__(), p, r, f1, a),
                  end='\r')

        print()
        all_scores = np.array(all_scores)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        self._save_if_better(save_best=save_best, force_checkpoint=force_checkpoint, score=f1)

        return all_scores, all_predictions, all_labels
