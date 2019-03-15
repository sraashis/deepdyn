"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os
import traceback

import torch
import torch.optim as optim

from ..probenet.model import UNet
from ..probenet.probenet_bee import ProbeNetBee
from ..probenet.probenet_dataloader import PatchesGenerator
from utils import auto_split as asp
from utils.measurements import ScoreAccumulator


def run(runs, transforms):
    for R in runs:
        for k, folder in R['Dirs'].items():
            os.makedirs(folder, exist_ok=True)

        R['acc'] = ScoreAccumulator()
        for split in os.listdir(R['Dirs']['splits_json']):
            splits = asp.load_split_json(os.path.join(R['Dirs']['splits_json'], split))

            R['checkpoint_file'] = split + '.tar'
            model = UNet(R['Params']['num_channels'], R['Params']['num_classes'])
            optimizer = optim.Adam(model.parameters(), lr=R['Params']['learning_rate'])
            if R['Params']['distribute']:
                model = torch.nn.DataParallel(model)
                model.float()
                optimizer = optim.Adam(model.module.parameters(), lr=R['Params']['learning_rate'])

            try:
                bee = ProbeNetBee(model=model, conf=R, optimizer=optimizer)
                if R.get('Params').get('mode') == 'train':
                    train_loader = PatchesGenerator.get_loader(conf=R, images=splits['train'], transforms=transforms,
                                                               mode='train')
                    val_loader = PatchesGenerator.get_loader_per_img(conf=R, images=splits['validation'],
                                                                     mode='validation', transforms=transforms)
                    bee.train(data_loader=train_loader, validation_loader=val_loader, epoch_run=bee.epoch_mse_loss)

                bee.resume_from_checkpoint(parallel_trained=R.get('Params').get('parallel_trained'))

                images = splits['test']
                test_loader = PatchesGenerator.get_loader_per_img(conf=R,
                                                                  images=images, mode='test', transforms=transforms)

                bee.test(data_loaders=test_loader, gen_images=True)
            except Exception as e:
                traceback.print_exc()

        print(R['acc'].get_prfa())
        f = open(R['Dirs']['logs'] + os.sep + 'score.txt', "w")
        f.write(', '.join(str(s) for s in R['acc'].get_prfa()))
        f.close()
