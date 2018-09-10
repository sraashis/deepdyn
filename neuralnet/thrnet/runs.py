import os

sep = os.sep

DRIVE = {
    'Params': {
        'num_channels': 1,
        'num_classes': 1,
        'batch_size': 32,
        'epochs': 200,
        'learning_rate': 0.001,
        'patch_shape': (27, 27),
        'patch_offset': (10, 10),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'checkpoint_file': 'THRNET-DRIVE.chk.tar',
        'log_frequency': 100,
        'validation_frequency': 4,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE_UNET_MAP' + sep + 'images',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'data' + sep + 'DRIVE_UNET_MAP' + sep + 'unet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
    }
}
