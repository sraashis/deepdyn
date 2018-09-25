import os

sep = os.sep

DRIVE = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'patch_shape': (16, 16),
        # 'patch_offset': (14, 14),
        'expand_patch_by': (0, 0),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': 'PATCHNET-DRIVE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE_UNET_MAP' + sep + 'images',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'data' + sep + 'DRIVE_UNET_MAP' + sep + 'patchnet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
    }
}
