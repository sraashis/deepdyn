import math
import os

import PIL.Image as IMG
import torch
import torch.nn.functional as F
import utils.img_utils as iu

from neuralnet.torchtrainer import NNTrainer
from neuralnet.utils.measurements import ScoreAccumulator
import numpy as np

sep = os.sep


class ThrnetTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')

    def train(self, optimizer=None, data_loader=None, validation_loader=None):

        if validation_loader is None:
            raise ValueError('Please provide validation loader.')

        logger = NNTrainer.get_logger(self.train_log_file,
                                      header='ID,EPOCH,BATCH,LOSS')

        val_logger = NNTrainer.get_logger(self.validation_log_file,
                                          header='ID,PRECISION,RECALL,F1,ACCURACY')
        print('Training...')
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            self._adjust_learning_rate(optimizer=optimizer, epoch=epoch)
            for i, data in enumerate(data_loader, 1):
                inputs = data['inputs'].to(self.device).float()
                y_thresholds = data['y_thresholds'].squeeze().to(self.device).float()

                optimizer.zero_grad()
                thr_map = self.model(inputs).squeeze()

                loss = F.mse_loss(thr_map, y_thresholds)
                loss.backward(retain_graph=True)
                optimizer.step()
                # if True:
                #     print(torch.cat([y_thresholds[..., None], thr_map[..., None]], 1))
                #     print('-------------------------------------------------')

                current_loss = math.sqrt(loss.item())
                running_loss += current_loss
                if i % self.log_frequency == 0:
                    print('Epochs[%d/%d] Batch[%d/%d] mse:%.5f' %
                          (
                              epoch, self.epochs, i, data_loader.__len__(), running_loss / self.log_frequency))
                    running_loss = 0.0

                self.flush(logger, ','.join(str(x) for x in [0, epoch, i, current_loss]))
            self.plot_train(file=self.train_log_file, batches_per_epochs=data_loader.__len__(), keys=['LOSS'])
            if epoch % self.validation_frequency == 0:
                self.evaluate(data_loaders=validation_loader, logger=val_logger, gen_images=False)
            self.plot_val(self.validation_log_file, batches_per_epoch=len(validation_loader))
        try:
            logger.close()
        except IOError:
            pass

    def evaluate(self, data_loaders=None, logger=None, gen_images=False):
        assert (logger is not None), 'Please Provide a logger'
        self.model.eval()

        print('\nEvaluating...')
        with torch.no_grad():
            eval_score = 0.0
            for loader in data_loaders:
                img_obj = loader.dataset.image_objects[0]

                segmented_img = torch.LongTensor(img_obj.working_arr.shape[0],
                                                 img_obj.working_arr.shape[1]).fill_(0).to(self.device)
                gt = torch.LongTensor(img_obj.ground_truth).to(self.device)
                img_loss = 0.0
                img_score = ScoreAccumulator()
                for i, data in enumerate(loader, 1):
                    inputs = data['inputs'].to(self.device).float()
                    prob_map = data['prob_map'].to(self.device).float()
                    y_thresholds = data['y_thresholds'].to(self.device).squeeze().float()
                    clip_ix = data['clip_ix'].to(self.device).int()

                    thr_map = self.model(inputs).squeeze()
                    # if True:
                    #     print(torch.cat(
                    #         [y_thresholds[..., None], thr_map[..., None], y_thresholds[..., None] - thr_map[..., None]],
                    #         1))
                    #     print('-------------------------------------------------')

                    loss = F.mse_loss(thr_map, y_thresholds)
                    current_loss = math.sqrt(loss.item())
                    img_loss += current_loss
                    thr = thr_map[..., None][..., None]
                    segmented = (prob_map > thr).long()
                    # Reconstruct the image
                    for j in range(segmented.shape[0]):
                        p, q, r, s = clip_ix[j]
                        segmented_img[p:q, r:s] += segmented[j]
                    print('Batch: ', i, end='\r')

                segmented_img[segmented_img > 0] = 255
                # img_score = ScoreAccumulator()
                img_loss = img_loss / i
                eval_score += img_loss
                if gen_images:
                    img = segmented_img.cpu().numpy()
                    img_score.add_array(img_obj.ground_truth, img)
                    img = iu.remove_connected_comp(np.array(img, dtype=np.uint8),
                                                   connected_comp_diam_limit=5)
                    IMG.fromarray(np.array(img, dtype=np.uint8)).save(
                        os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + '.png'))
                else:
                    img_score.add_tensor(segmented_img, gt)

                prf1a = img_score.get_prf1a()
                print(img_obj.file_name, ' PRF1A', prf1a, ' loss: ' + str(eval_score / len(data_loaders)))
                self.flush(logger, ','.join(str(x) for x in [img_obj.file_name] + prf1a))

            self._save_if_better(score=len(data_loaders) / eval_score)
