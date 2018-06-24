import os

import PIL.Image as IMG
import numpy as np
import torch

import neuralnet.unet.utils as ut
from neuralnet.torchtrainer import NNTrainer
from neuralnet.utils.measurements import ScoreAccumulator

sep = os.sep


class UNetNNTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)

    def evaluate(self, data_loader=None, force_checkpoint=False, mode=None, **kwargs):

        assert (mode == 'eval' or mode == 'train'), 'Mode can either be eval or train'
        to_dir = kwargs['segmented_out'] if 'segmented_out' in kwargs else None
        patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else None

        self.model.eval()
        data_loader = data_loader if isinstance(data_loader, list) else [data_loader]
        score_acc = ScoreAccumulator()
        print('\nEvaluating...')
        with torch.no_grad():
            for loader in data_loader:
                if mode is 'train':
                    score_acc.accumulate(self._evaluate(data_loader=loader,
                                                        force_checkpoint=force_checkpoint, mode=mode))
                if mode is 'eval' and to_dir is not None:
                    scores, predictions, labels = self._evaluate(data_loader=loader,
                                                                 force_checkpoint=force_checkpoint, mode=mode)
                    segmented = ut.merge_patches(scores, loader.dataset.image_objects[0].working_arr.shape, patch_size)
                    IMG.fromarray(segmented).save(to_dir + sep + loader.dataset.image_objects[0].file_name + '.png')
        if mode is 'train':
            self._save_if_better(force_checkpoint=force_checkpoint, score=score_acc.get_prf1a()[2])

    def _evaluate(self, data_loader=None, force_checkpoint=False, mode=None):

        assert (mode == 'eval' or mode == 'train'), 'Mode can either be eval or train'
        score_acc = ScoreAccumulator()
        all_predictions = []
        all_scores = []
        all_labels = []
        for i, data in enumerate(data_loader, 0):
            ID, inputs, labels = data[0], data[1].to(self.device), data[2].to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Accumulate scores
            if mode is 'eval':
                all_scores += outputs.clone().cpu().numpy().tolist()
                all_predictions += predicted.clone().cpu().numpy().tolist()
                all_labels += labels.clone().cpu().numpy().tolist()

            p, r, f1, a = score_acc.add(labels, predicted).get_prf1a()
            print('Batch[%d/%d] pre:%.3f rec:%.3f f1:%.3f acc:%.3f' % (i + 1, data_loader.__len__(), p, r, f1, a),
                  end='\r')
        self._log(','.join(str(x) for x in
                           [data_loader.dataset.image_objects[0].file_name, 1, self.checkpoint['epochs'],
                            0] + score_acc.get_prf1a()))
        print()
        if mode is 'eval':
            all_scores = np.array(all_scores)
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
        if mode is 'train':
            return score_acc
        return all_scores, all_predictions, all_labels
