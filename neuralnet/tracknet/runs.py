import os

sep = os.sep

DRIVE = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 16,
        'epochs': 1,
        'learning_rate': 0.01,
        'patch_shape': (15, 15),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'test',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE-TRACKNET' + sep + 'mats',
        'mask': 'data' + sep + 'DRIVE-TRACKNET' + sep + 'mask',
        'logs': 'data' + sep + 'DRIVE-TRACKNET' + sep + 'tracknet_logs',
        'splits_json': 'data' + sep + 'DRIVE-TRACKNET' + sep + 'tracknet_splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('.')[0].split('_')[-1] + '_test_mask.gif'
    }
}
