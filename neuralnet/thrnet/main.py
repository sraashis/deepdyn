import os
import sys
import traceback

try:
    BASE_PROJECT_DIR = '/home/ak/PycharmProjects/ature'
    sys.path.append(BASE_PROJECT_DIR)
    os.chdir(BASE_PROJECT_DIR)
except:
    BASE_PROJECT_DIR = '/home/akhanal1/ature'
    sys.path.append(BASE_PROJECT_DIR)
    os.chdir(BASE_PROJECT_DIR)

import torch
import torch.optim as optim
from neuralnet.thrnet.inception import InceptionThrNet
from neuralnet.thrnet.thrnet_dataloader import PatchesGenerator
from neuralnet.thrnet.thrnet_trainer import ThrnetTrainer
import torchvision.transforms as transforms
from neuralnet.utils import auto_split as asp
from neuralnet.thrnet.runs import DRIVE32a

# RUNS = [DRIVE32, DRIVE16]

RUNS = [DRIVE32a]
torch.cuda.set_device(1)

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    for R in RUNS:
        for k, folder in R['Dirs'].items():
            os.makedirs(folder, exist_ok=True)

        splits = asp.create_split_json(
            images_src_dir=R.get('Dirs').get('image'),
            to_file=os.path.join(R.get('Dirs').get('logs'), R.get('Params').get('checkpoint_file') + '.json'))
        patch_shape = R['Params']['patch_shape'][0] + R['Params']['expand_patch_by'][0]
        model = InceptionThrNet(patch_shape, R['Params']['num_channels'], R['Params']['num_classes'])
        optimizer = optim.Adam(model.parameters(), lr=R['Params']['learning_rate'])
        if R['Params']['distribute']:
            model = torch.nn.DataParallel(model)
            model.float()
            optimizer = optim.Adam(model.module.parameters(), lr=R['Params']['learning_rate'])

        try:
            drive_trainer = ThrnetTrainer(model=model, run_conf=R)

            if R.get('Params').get('mode') == 'train':
                train_loader = PatchesGenerator.get_loader(run_conf=R, images=splits['train'], transforms=transform,
                                                           mode='train')
                val_loader = PatchesGenerator.get_loader_per_img(run_conf=R, images=splits['validation'],
                                                                 mode='validation')
                drive_trainer.train(optimizer=optimizer, data_loader=train_loader, validation_loader=val_loader)

            drive_trainer.resume_from_checkpoint(parallel_trained=R.get('Params').get('parallel_trained'))
            test_loader = PatchesGenerator.get_loader_per_img(run_conf=R, images=splits['test'], mode='test')

            log_file = os.path.join(R['Dirs']['logs'], R['Params']['checkpoint_file'] + '-TEST.csv')
            logger = drive_trainer.get_logger(log_file, header='ID,TYPE,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS')
            drive_trainer.evaluate(data_loaders=test_loader, logger=logger, gen_images=True)
            logger.close()
        except Exception as e:
            traceback.print_exc()
