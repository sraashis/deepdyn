"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import os
import traceback

import torch
import torch.optim as optim
import torchvision.transforms as transforms

import neuralnet.unet.runs as rs
from neuralnet.unet.model import UNet
from neuralnet.unet.unet_dataloader import PatchesGenerator
from neuralnet.unet.unet_trainer import UNetBee
from nnbee.utils import auto_split as asp
from nnbee.utils.measurements import ScoreAccumulator

# RUNS1 = [rs.DRIVE1, rs.DRIVE2,
#          rs.STARE1, rs.STARE2,
#          rs.WIDE1, rs.WIDE2,
#          rs.VEVIO1, rs.VEVIO2]
#
# RUNS2 = [rs.DRIVE, rs.DRIVE3,
#          rs.STARE, rs.STARE3,
#          rs.WIDE, rs.WIDE3,
#          rs.VEVIO, rs.VEVIO3]

RUNS = [rs.DRIVE]


def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    for R in RUNS:
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
                drive_trainer = UNetBee(model=model, conf=R, optimizer=optimizer)
                if R.get('Params').get('mode') == 'train':
                    train_loader = PatchesGenerator.get_loader(conf=R, images=splits['train'], transforms=transform,
                                                               mode='train')
                    val_loader = PatchesGenerator.get_loader_per_img(conf=R, images=splits['validation'],
                                                                     mode='validation')
                    drive_trainer.train(data_loader=train_loader, validation_loader=val_loader,
                                        epoch_run=drive_trainer.epoch_ce_loss)

                drive_trainer.resume_from_checkpoint(parallel_trained=R.get('Params').get('parallel_trained'))

                test_loader = PatchesGenerator.get_loader_per_img(conf=R,
                                                                  images=splits['test'], mode='test')
                drive_trainer.test(test_loader)
            except Exception as e:
                traceback.print_exc()

        print(R['acc'].get_prfa())
        f = open(R['Dirs']['logs'] + os.sep + 'score.txt', "w")
        f.write(', '.join(str(s) for s in R['acc'].get_prfa()))
        f.close()


if __name__ == "__main__":
    main()
