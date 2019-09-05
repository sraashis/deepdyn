"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os
import traceback

import torch
import torch.optim as optim

from utils import auto_split as asp
from utils.measurements import ScoreAccumulator
from ..unet.model import UNet
from ..unet.unet_dataloader import PatchesGenerator
from ..unet.unet_trainer import UNetTrainer


def run(runs, transforms):
    for R in runs:
        for k, folder in R['Dirs'].items():
            os.makedirs(folder, exist_ok=True)
        R['acc'] = ScoreAccumulator()
        for split_file in os.listdir(R['Dirs']['splits_json']):
            splits = asp.load_split_json(os.path.join(R['Dirs']['splits_json'], split_file))

            R['checkpoint_file'] = split_file + '.tar'
            model = UNet(R['Params']['num_channels'], R['Params']['num_classes'])
            optimizer = optim.Adam(model.parameters(), lr=R['Params']['learning_rate'])
            if R['Params']['distribute']:
                model = torch.nn.DataParallel(model)
                model.float()
                optimizer = optim.Adam(model.module.parameters(), lr=R['Params']['learning_rate'])

            try:
                trainer = UNetTrainer(model=model, conf=R, optimizer=optimizer)
                if R.get('Params').get('mode') == 'train':
                    # train_loader, val_loader = PatchesGenerator.random_split(conf=R,
                    #                                                          images=splits['train'] + splits[
                    #                                                              'validation'],
                    #                                                          transforms=transforms, mode='train')

                    train_loader = PatchesGenerator.get_loader(conf=R, images=splits['train'], transforms=transforms,
                                                               mode='train')
                    val_loader = PatchesGenerator.get_loader_per_img(conf=R, images=splits['validation'],
                                                                     mode='validation', transforms=transforms)

                    # print('### Train Val Batch size:', len(train_loader.dataset), len(val_loader.dataset))
                    trainer.train(data_loader=train_loader, validation_loader=val_loader,
                                  epoch_run=trainer.epoch_ce_loss)

                test_loader = PatchesGenerator.get_loader_per_img(conf=R,
                                                                  images=splits['test'], mode='test',
                                                                  transforms=transforms)

                trainer.resume_from_checkpoint(parallel_trained=R.get('Params').get('parallel_trained'))
                trainer.test(test_loader)
            except Exception as e:
                traceback.print_exc()

        print(R['acc'].get_prfa())
        f = open(R['Dirs']['logs'] + os.sep + 'score.txt', "w")
        f.write(', '.join(str(s) for s in R['acc'].get_prfa()))
        f.close()
