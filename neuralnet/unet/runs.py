import os

import data_access

sep = os.sep

DRIVE = {

    'P': {
        'num_channels': 1,
        'num_classes': 1,
        'batch_size': 8,
        'epochs': 100,
        'learning_rate': 0.0001,
        'patch_shape': (388, 388),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'checkpoint_file': 'UNET-DRIVE.chk.tar',
        'mode': 'train'
    },

    'D': data_access.Drive_Dirs,
    'F': data_access.Drive_Funcs
}
