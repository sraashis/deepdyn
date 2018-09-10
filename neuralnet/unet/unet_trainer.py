import os

import numpy as np
import torch
from PIL import Image as IMG

import utils.img_utils as imgutils
from neuralnet.torchtrainer import NNTrainer
from neuralnet.utils.measurements import ScoreAccumulator

sep = os.sep


class UNetNNTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')

    def evaluate(self, data_loaders=None, force_checkpoint=False, logger=None, mode=None):
        assert (logger is not None), 'Please Provide a logger'
        self.model.eval()

        print('\nEvaluating...')
        with torch.no_grad():
            eval_score = ScoreAccumulator()
            for loader in data_loaders:
                img_obj = loader.dataset.image_objects[0]
                segmented_map, labels_acc = [], []

                img_score = ScoreAccumulator()
                for i, data in enumerate(loader, 1):
                    inputs, labels = data['inputs'].to(self.device), data['labels'].to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)

                    current_score = ScoreAccumulator()
                    current_score.add_tensor(labels, predicted)
                    img_score.accumulate(current_score)
                    eval_score.accumulate(current_score)

                    if mode is 'test':
                        segmented_map += outputs.clone().cpu().numpy().tolist()
                        labels_acc += labels.clone().cpu().numpy().tolist()

                    self.flush(logger, ','.join(
                        str(x) for x in
                        [img_obj.file_name, 1, self.checkpoint['epochs'], 0] + current_score.get_prf1a()))

                print(img_obj.file_name + ' PRF1A: ', img_score.get_prf1a())
                if mode is 'test':
                    segmented_map = np.exp(np.array(segmented_map)[:, 1, :, :]).squeeze()
                    segmented_map = np.array(segmented_map * 255, dtype=np.uint8)
                    # labels_acc = np.array(np.array(labels_acc).squeeze()*255, dtype=np.uint8)

                    maps_img = imgutils.merge_patches(patches=segmented_map, image_size=img_obj.working_arr.shape,
                                                      patch_size=self.patch_shape,
                                                      offset_row_col=self.patch_offset)
                    IMG.fromarray(maps_img).save(os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + '.png'))

        if mode is 'train':
            self._save_if_better(force_checkpoint=force_checkpoint, score=eval_score.get_prf1a()[2])
