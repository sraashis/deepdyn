import math
import os

import PIL.Image as IMG
import torch
import torch.nn.functional as F

from neuralnet.torchtrainer import NNTrainer
import numpy as np
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
                inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device)

                optimizer.zero_grad()
                thr_map = self.model(inputs).squeeze()

                # if True:
                #     print(torch.cat([labels[..., None].squeeze(), thr_map[..., None].squeeze()], 1))
                #     print('-------------------------------------------------')
                # print('thr_map', thr_map)
                # print('labels', labels)
                # print('thr_map typppeeeee', type(thr_map), type(labels))

                loss = F.mse_loss(thr_map, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                current_loss = math.sqrt(loss.item())

                # result = 1 - spatial.distance.cosine(thr_map, labels)
                # print('resullllt', result)

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
            eval_score = 0.000001
            for loader in data_loaders:
                img_obj = loader.dataset.image_objects[0]

                segmented_img = torch.LongTensor(img_obj.working_arr.shape[0],
                                                 img_obj.working_arr.shape[1], 3).fill_(0).to(self.device)
                img_loss = 0.000001
                for i, data in enumerate(loader, 1):
                    inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device)

                    positions = data['POS'].to(self.device)
                    outputs = self.model(inputs).squeeze()


                    # print('thr_map labels Evaluation', outputs, 'labels', labels)
                    # print('thr_map typppeeeee', type(outputs), type(labels))

                    predicted = outputs + positions.float()
                    loss = F.mse_loss(outputs.squeeze(), labels.squeeze())
                    current_loss = math.sqrt(loss.item())
                    img_loss += current_loss
                    if len(outputs.shape) == 1:
                        outputs = outputs[None, ...]

                    labels = labels + positions.float()
                    for j in range(outputs.shape[0]):
                        x, y = int(labels[j][0]), int(labels[j][1])
                        x_pred, y_pred = int(predicted[j][0]), int(predicted[j][1])
                        segmented_img[:, :, :][x, y] = 255
                        segmented_img[:, :, 0][x_pred, y_pred] = 255
                    self.flush(logger,
                               ','.join(str(x) for x in [img_obj.file_name] + [current_loss]))

                img_loss = img_loss / loader.__len__()
                print(img_obj.file_name + ' loss: ' + str(img_loss))
                eval_score += img_loss

                segmented_img[segmented_img > 0] = 255
                if gen_images:
                    img = segmented_img.cpu().numpy()
                    IMG.fromarray(np.array(img, np.uint8)).save(
                        os.path.join(self.log_dir, str(self.log_counter) + img_obj.file_name.split('.')[0] + '.png'))
                    self.log_counter += 1

            self._save_if_better(score=len(data_loaders) / eval_score)
