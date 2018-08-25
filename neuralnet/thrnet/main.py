BASE_PROJECT_DIR = '/home/akhanal1/ature'
# BASE_PROJECT_DIR = '/home/ak/PycharmProjects/ature'

import os
import sys
import traceback

sys.path.append(BASE_PROJECT_DIR)
os.chdir(BASE_PROJECT_DIR)

import torch
import torch.optim as optim
from neuralnet.thrnet.model import ThrNet
from neuralnet.thrnet.thrnet_dataloader import PatchesGenerator, get_loader_per_img
from neuralnet.thrnet.thrnet_trainer import ThrnetTrainer
import torchvision.transforms as transforms
from neuralnet.thrnet.runs import DRIVE

RUNS = [DRIVE]

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    for I in RUNS:
        for k, folder in I['D'].items():
            os.makedirs(folder, exist_ok=True)
        model = ThrNet(I['P']['patch_shape'][0], I['P']['num_channels'])
        optimizer = optim.Adam(model.parameters(), lr=I['P']['learning_rate'])
        if I['P']['distribute']:
            model = torch.nn.DataParallel(model)
            model.float()
            optimizer = optim.Adam(model.module.parameters(), lr=I['P']['learning_rate'])

        try:
            drive_trainer = ThrnetTrainer(model=model,
                                          checkpoint_file=I['P']['checkpoint_file'],
                                          log_file=I['P']['checkpoint_file'] + '.csv',
                                          use_gpu=I['P']['use_gpu'])
            if I['P']['mode'] == 'train':
                train_loader = PatchesGenerator(
                    images_dir=I['D']['train_img'],
                    mask_dir=I['D']['train_mask'],
                    manual_dir=I['D']['train_manual'],
                    transforms=transform,
                    get_mask=I['F']['train_mask_getter'],
                    get_truth=I['F']['train_gt_getter'],
                    patch_shape=I['P']['patch_shape'],
                    offset_shape=(15, 15)
                ).get_loader(batch_size=I['P']['batch_size'])

                val_loaders = get_loader_per_img(
                    images_dir=I['D']['val_img'],
                    mask_dir=I['D']['val_mask'],
                    manual_dir=I['D']['val_manual'],
                    transforms=transform,
                    get_mask=I['F']['val_mask_getter'],
                    get_truth=I['F']['val_gt_getter'],
                    patch_shape=I['P']['patch_shape']
                )

                drive_trainer.train(optimizer=optimizer,
                                    data_loader=train_loader,
                                    epochs=I['P']['epochs'],
                                    validation_loader=val_loaders,
                                    force_checkpoint=False, log_frequency=50)
            else:
                # drive_trainer.resume_from_checkpoint(parallel_trained=False)
                pass

            test_loaders = get_loader_per_img(
                images_dir=I['D']['test_img'],
                mask_dir=I['D']['test_mask'],
                manual_dir=I['D']['test_manual'],
                transforms=transform,
                get_mask=I['F']['test_mask_getter'],
                get_truth=I['F']['test_gt_getter'],
                patch_shape=I['P']['patch_shape']
            )

            logger = drive_trainer.get_logger(I['P']['checkpoint_file'] + '-TEST.csv')
            drive_trainer.evaluate(data_loader=test_loaders, mode='eval', patch_size=I['P']['patch_shape'],
                                   segmented_out=I['D']['test_img_out'],
                                   logger=logger)
            logger.close()
        except Exception as e:
            traceback.print_exc()
