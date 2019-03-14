"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as IMG

from nnbee.torchtrainer import NNBee
from nnbee.utils.measurements import ScoreAccumulator

sep = os.sep


class ProbeNetBee(NNBee):
    def __init__(self, **kwargs):
        NNBee.__init__(self, **kwargs)
        self.patch_shape = self.conf.get('Params').get('patch_shape')
        self.patch_offset = self.conf.get('Params').get('patch_offset')
        self.dparm = self.conf.get("Funcs").get('dparm')

    # This method should work invariant to input/output channels
    def _eval(self, data_loaders=None, logger=None, gen_images=False, score_acc=None):
        assert isinstance(score_acc, ScoreAccumulator)
        with torch.no_grad():
            for loader in data_loaders:
                img_obj = loader.dataset.image_objects[0]

                if len(img_obj.working_arr.shape) == 3:
                    x, y, c = img_obj.working_arr.shape

                elif len(img_obj.working_arr.shape) == 2:
                    (x, y), c = img_obj.working_arr.shape, 1

                map_img = torch.FloatTensor(c, x, y).fill_(0).to(self.device)

                img_loss = 0.0
                for i, data in enumerate(loader, 1):
                    inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).float()
                    clip_ix = data['clip_ix'].to(self.device).int()

                    outputs = self.model(inputs)
                    loss = F.mse_loss(outputs, labels).item()

                    img_loss += loss

                    for j in range(outputs.shape[0]):
                        p, q, r, s = clip_ix[j]
                        map_img[:, p:q, r:s] = outputs[j, :, :, :]

                    print('Batch: ', i, end='\r')

                if gen_images:
                    map_img = map_img.cpu().numpy().squeeze()

                    #  Dimension of tensor and PIL image are reverted. We need to fix that before saving PIL image
                    if len(map_img.shape) == 3:
                        map_img = np.rollaxis(map_img, 0, 3)

                    IMG.fromarray(np.array(map_img, dtype=np.uint8)).save(
                        os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + '.png'))
                else:
                    score_acc += img_loss / loader.__len__()
                print('\n' + img_obj.file_name, ' Image LOSS: ', img_loss / loader.__len__())

                self.flush(logger, ','.join(str(x) for x in [img_obj.file_name, img_loss / loader.__len__()]))

        self._save_if_better(score=len(data_loaders) / score_acc)
