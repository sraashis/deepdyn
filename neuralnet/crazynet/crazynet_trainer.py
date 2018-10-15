import os

import PIL.Image as IMG
import numpy as np
import torch

import utils.img_utils as iu
import torch.nn.functional as F
from neuralnet.torchtrainer import NNTrainer
from neuralnet.utils.measurements import ScoreAccumulator
import math

sep = os.sep


class ThrnetTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')

    def evaluate(self, data_loaders=None, logger=None, gen_images=False):
        assert (logger is not None), 'Please Provide a logger'
        self.model.eval()

        print('\nEvaluating...')
        with torch.no_grad():
            eval_score = 0.0

            for loader in data_loaders:
                img_obj = loader.dataset.image_objects[0]
                x, y = img_obj.working_arr.shape[0], img_obj.working_arr.shape[1]
                # predicted_img = torch.FloatTensor(x, y).fill_(0).to(self.device)
                map_img = torch.FloatTensor(x, y).fill_(0).to(self.device)

                gt = torch.FloatTensor(img_obj.ground_truth).to(self.device)

                for i, data in enumerate(loader, 1):
                    inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).float()
                    clip_ix = data['clip_ix'].to(self.device).int()

                    outputs = self.model(inputs)
                    loss = F.mse_loss(outputs.squeeze(), labels.squeeze())
                    eval_score += math.sqrt(loss.item())
                    # _, predicted = torch.max(outputs, 1)
                    # predicted_map = outputs[:, 1, :, :]

                    for j in range(outputs.shape[0]):
                        p, q, r, s = clip_ix[j]
                        # predicted_img[p:q, r:s] = predicted[j]
                        map_img[p:q, r:s] = outputs[j]
                    print('Batch: ', i, end='\r')
                eval_score = eval_score/i
                img_score = ScoreAccumulator()
                # map_img = torch.exp(map_img) * 255
                # predicted_img = predicted_img * 255

                if gen_images:
                    map_img = map_img.cpu().numpy()
                    # predicted_img = predicted_img.cpu().numpy()
                    # img_score.add_array(img_obj.ground_truth, predicted_img)
                    # predicted_img = iu.remove_connected_comp(np.array(predicted_img, dtype=np.uint8),
                    #                                          connected_comp_diam_limit=10)
                    # IMG.fromarray(predicted_img.astype(np.uint8)).save(
                    #     os.path.join(self.log_dir, 'pred_' + img_obj.file_name.split('.')[0] + '.png'))
                    IMG.fromarray(np.array(map_img, dtype=np.uint8)).save(
                        os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + '.png'))
                else:
                    # img_score.add_tensor(predicted_img, gt)
                    eval_score += img_score.get_prf1a()[2]

                prf1a = img_score.get_prf1a()
                print(img_obj.file_name, ' PRF1A', eval_score / len(data_loaders))
                self.flush(logger, ','.join(str(x) for x in [img_obj.file_name] + prf1a))

        self._save_if_better(score= len(data_loaders)/ eval_score)

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
                inputs, labels = data['inputs'].to(self.device), data['labels'].float().squeeze().to(
                    self.device)

                optimizer.zero_grad()
                thr_map = self.model(inputs).squeeze()

                loss = F.mse_loss(thr_map, labels)
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

