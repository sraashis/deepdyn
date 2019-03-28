import math
import os

import PIL.Image as IMG
import torch
import torch.nn.functional as F

from neuralnet.torchtrainer import NNTrainer
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

plt.switch_backend('agg')
from scipy import spatial

sep = os.sep


class TracknetTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')
        self.log_counter = 0

    def train(self, optimizer=None, data_loader=None, validation_loader=None):

        if validation_loader is None:
            raise ValueError('Please provide validation loader.')

        logger = NNTrainer.get_logger(self.train_log_file,
                                      header='ID,EPOCH,BATCH,LOSS')

        val_logger = NNTrainer.get_logger(self.validation_log_file,
                                          header='ID,LOSS')
        print('Training...')
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0

            self._adjust_learning_rate(optimizer=optimizer, epoch=epoch)
            for i, data in enumerate(data_loader, 1):
                inputs, rho, labels = data['inputs'].to(self.device).float(), \
                                      data['rho'].to(self.device).float().cuda().squeeze(), \
                                      data['labels'].to(self.device)

                optimizer.zero_grad()
                thr_map = self.model(inputs).squeeze()

                labels = labels.squeeze()
                thr_map = thr_map.squeeze()
                diff = thr_map - labels
                diff[diff > 180] -= 360
                loss = torch.abs(diff).mean()
                loss.backward()
                optimizer.step()
                current_loss = loss.item()

                running_loss += current_loss
                if i % self.log_frequency == 0:
                    print('Epochs[%d/%d] Batch[%d/%d] mse:%.5f' %
                          (
                              epoch, self.epochs, i, data_loader.__len__(), running_loss / self.log_frequency))
                    running_loss = 0.0

                self.flush(logger, ','.join(str(x) for x in [0, epoch, i, current_loss]))
            self.plot_train(file=self.train_log_file, batches_per_epochs=data_loader.__len__(), keys=['LOSS'])
            if epoch % self.validation_frequency == 0:
                self.checkpoint['epochs'] = epoch
                self.evaluate(data_loaders=validation_loader, logger=val_logger, gen_images=False)
            self.plot_val(self.validation_log_file, batches_per_epoch=len(validation_loader))
        # print('maxloss', maxloss)

        try:
            logger.close()
        except IOError:
            pass

    def evaluate(self, data_loaders=None, logger=None, gen_images=False):
        assert (logger is not None), 'Please Provide a logger'
        self.model.eval()

        print('\nEvaluating...')
        keys = range(91)
        angloss = {el: [] for el in keys}
        labelcount = {el: 0 for el in keys}
        with torch.no_grad():
            eval_score = 0.000001
            for loader in data_loaders:
                img_obj = loader.dataset.image_objects[0]

                segmented_img = torch.LongTensor(img_obj.working_arr.shape[0],
                                                 img_obj.working_arr.shape[1], 3).fill_(0).to(self.device)
                # segmented_img[:, :, 1] = torch.LongTensor(img_obj.working_arr).to(self.device)
                img_loss = 0.000001
                for i, data in enumerate(loader, 1):
                    inputs, rho, labels = data['inputs'].to(self.device).float(), \
                                          data['rho'].to(self.device).float().cuda().squeeze(), \
                                          data['labels'].to(self.device).squeeze()
                    IJs = data['POS'].to(self.device).long()

                    # positions = data['POS'].to(self.device)
                    # prev_positions = data['PREV'].to(self.device)
                    outputs = self.model(inputs).squeeze()

                    diff = torch.abs(outputs - labels)
                    diff[diff > 180] -= 360
                    diff = abs(diff)
                    if True:
                        for e, l in zip(diff.cpu().numpy(), labels.cpu().numpy()):
                            angloss[int(l)].append(e)
                            labelcount[int(l)] += 1
                        # print(torch.cat([labels[..., None], outputs[..., None], labels[..., None] - outputs[..., None]], 1))

                    loss = diff.abs().mean()
                    current_loss = loss.item()
                    # print('current_loss', current_loss)
                    img_loss += current_loss

                    # if len(outputs.shape) == 1:
                    #     outputs = outputs[None, ...]
                    for j in range(outputs.shape[0]):
                        x, y = int(IJs[j][0]), int(IJs[j][1])
                        segmented_img[:, :, 0][x, y] = 255
                        x_pred, y_pred = int(np.asarray(rho[j]) * np.cos(outputs[j])), int(
                            np.asarray(rho[j]) * np.sin(outputs[j]))
                        segmented_img[:, :, 1][x + x_pred, y + y_pred] = 255

                    self.flush(logger,
                               ','.join(str(x) for x in [img_obj.file_name] + [current_loss]))

                img_loss = img_loss / loader.__len__()
                eval_score += img_loss

                if gen_images:
                    img = segmented_img.cpu().numpy()
                    IMG.fromarray(np.array(img, np.uint8)).save(
                        os.path.join(self.log_dir, str(self.log_counter) + img_obj.file_name.split('.')[0] + '.png'))
                    self.log_counter += 1

            self._save_if_better(score=len(data_loaders) / eval_score)

            plt.rcParams["figure.figsize"] = (24, 16)

            err = list(angloss.values())
            # count = list(labelcount.values())
            # res = np.zeros(len(keys))
            # for k in keys:
            #     if count[k] == 0:
            #         continue
            #     res[k] = err[k] / count[k]
            res = []
            temp = []
            for i in range(len(err)):
                if (i+1) % 5 == 0:
                    if i > 0:
                        res.append(temp)
                    temp = []
                else:
                    temp += err[i]

            plt.boxplot(res)
            plt.savefig('error_distrib.png')
