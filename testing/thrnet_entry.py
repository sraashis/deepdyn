import os
import sys
import traceback

sys.path.append('/home/akhanal1/ature')
os.chdir('/home/akhanal1/ature')

import torch
import torch.optim as optim
from neuralnet.thrnet.thrnet import ThrNet
from neuralnet.thrnet.thrnet_dataloader import split_drive_dataset
from neuralnet.thrnet.thrnet_trainer import ThrnetTrainer
import torchvision.transforms as transforms

if __name__ == "__main__":
    sep = os.sep
    Params = {}
    Params['num_channels'] = 1
    Params['batch_size'] = 32
    Params['num_classes'] = 1
    Params['epochs'] = 5
    Params['patch_size'] = (31, 31)  # rows X cols
    Params['use_gpu'] = True
    Params['learning_rate'] = 0.001
    Params['distribute'] = True

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    # ---------------------------------------------------------------------------------------------------------
    # Define the network
    model = ThrNet(Params['patch_size'][0], Params['num_channels'])
    # exit(0)
    optimizer = optim.Adam(model.parameters(), lr=Params['learning_rate'])
    if Params['distribute']:
        model = torch.nn.DataParallel(model)
        model.float()
        optimizer = optim.Adam(model.module.parameters(), lr=Params['learning_rate'])

    """
    ################## UNET Drive Data set ################
    """
    try:
        Dirs = {}
        Dirs['train'] = 'data' + sep + 'DRIVE' + sep + 'thr_training'
        Dirs['test'] = 'data' + sep + 'DRIVE' + sep + 'thr_testing'
        Dirs['segmented'] = 'data' + sep + 'DRIVE' + sep + 'segmented_thr'

        checkpoint = 'unet-DRIVE-THR.chk.tar'
        drive_trainer = ThrnetTrainer(model=model,
                                      checkpoint_file=checkpoint,
                                      log_file=checkpoint + '.csv',
                                      use_gpu=Params['use_gpu'])
        train_loader, val_loader, test_loader = split_drive_dataset(Dirs=Dirs, transform=transform,
                                                                    batch_size=Params['batch_size'],
                                                                    patch_shape=Params['patch_size'])
        # drive_trainer.resume_from_checkpoint(parallel_trained=False)
        drive_trainer.train(optimizer=optimizer,
                            data_loader=train_loader,
                            epochs=Params['epochs'],
                            validation_loader=val_loader,
                            force_checkpoint=True, log_frequency=20)
        drive_trainer.resume_from_checkpoint(parallel_trained=False)
        logger = drive_trainer.get_logger(checkpoint + '-TEST_THR.csv')
        drive_trainer.evaluate(data_loader=test_loader, mode='eval', patch_size=Params['patch_size'],
                               segmented_out=Dirs['segmented'],
                               logger=logger)
        logger.close()
    except Exception as e:
        traceback.print_exc()
    # End
