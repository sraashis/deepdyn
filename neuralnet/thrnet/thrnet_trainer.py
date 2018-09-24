import os

import PIL.Image as IMG
import numpy as np
import torch

from neuralnet.torchtrainer import NNTrainer
from neuralnet.utils.measurements import ScoreAccumulator

sep = os.sep


class ThrnetTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')

    # def train(self, optimizer=None, data_loader=None, validation_loader=None):
    #
    #     if validation_loader is None:
    #         raise ValueError('Please provide validation loader.')
    #
    #     logger = NNTrainer.get_logger(self.log_file, 'ID,TYPE,EPOCH,BATCH,LOSS')
    #     print('Training...')
    #     for epoch in range(1, self.epochs + 1):
    #         self.model.train()
    #         running_loss = 0.0
    #         self.adjust_learning_rate(optimizer=optimizer, epoch=epoch)
    #         for i, data in enumerate(data_loader, 1):
    #             inputs, y_thresholds = data['inputs'].to(self.device), data['y_thresholds'].float().to(self.device)
    #
    #             optimizer.zero_grad()
    #             thr_map = self.model(inputs)
    #
    #             # if True:
    #             #     print(torch.cat([y_thresholds[..., None], thr_map], 1))
    #             #     print('-------------------------------------------------')
    #
    #             y_thresholds = y_thresholds.squeeze()
    #             thr_map = thr_map.squeeze()
    #             loss = F.mse_loss(thr_map, y_thresholds)
    #             loss.backward()
    #             optimizer.step()
    #
    #             current_loss = math.sqrt(loss.item())
    #             running_loss += current_loss
    #             if i % self.log_frequency == 0:
    #                 print('Epochs[%d/%d] Batch[%d/%d] mse:%.5f' %
    #                       (
    #                           epoch, self.epochs, i, data_loader.__len__(), running_loss / self.log_frequency))
    #                 running_loss = 0.0
    #
    #             self.flush(logger, ','.join(str(x) for x in [0, 0, epoch, i, current_loss]))
    #
    #         if epoch % self.validation_frequency == 0:
    #             self.evaluate(data_loaders=validation_loader, logger=logger,
    #                           mode='test')
    #     try:
    #         logger.close()
    #     except IOError:
    #         pass

    def evaluate(self, data_loaders=None, logger=None, epoch=0):
        assert (logger is not None), 'Please Provide a logger'
        self.model.eval()

        print('\nEvaluating...')
        with torch.no_grad():
            eval_score = 0.0

            for loader in data_loaders:
                img_obj = loader.dataset.image_objects[0]
                segmented_img = torch.cuda.LongTensor(img_obj.working_arr.shape[0],
                                                      img_obj.working_arr.shape[1]).fill_(0).to(self.device)
                gt = torch.LongTensor(img_obj.ground_truth).to(self.device)

                for i, data in enumerate(loader, 1):
                    inputs, labels = data['inputs'].float().to(self.device), data['labels'].float().to(self.device)
                    clip_ix = data['clip_ix'].int().to(self.device)

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)

                    for j in range(predicted.shape[0]):
                        p, q, r, s = clip_ix[j]
                        segmented_img[p:q, r:s] += predicted[j]
                    print('Batch: ', i, end='\r')

                segmented_img[segmented_img > 0] = 255
                # segmented_img[img_obj.mask == 0] = 0

                img_score = ScoreAccumulator()

                if self.model.training:
                    img_score.add_tensor(segmented_img, gt)
                else:
                    segmented_img = segmented_img.cpu().numpy()
                    img_score.add_array(img_obj.ground_truth, segmented_img)
                    eval_score += img_score.get_prf1a()[2]
                    IMG.fromarray(np.array(segmented_img, dtype=np.uint8)).save(
                        os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + '.png'))

                print(img_obj.file_name, ' PRF1A', img_score.get_prf1a())

        self._save_if_better(score=eval_score / len(data_loaders))
