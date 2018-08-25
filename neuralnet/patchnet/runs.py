import os
import data_access

sep = os.sep

DRIVE = {

    'P': {
        'num_channels': 1,
        'num_classes': 1,
        'batch_size': 48,
        'epochs': 100,
        'learning_rate': 0.001,
        'patch_shape': (31, 31),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'checkpoint_file': 'PATCHNET-DRIVE.chk.tar',
        'mode': 'train'
    },
    'D': data_access.Drive_Dirs,
    'F': data_access.Drive_Funcs
}
