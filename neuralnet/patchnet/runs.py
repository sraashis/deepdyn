import os

sep = os.sep

DRIVE = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 32,
        'epochs': 30,
        'learning_rate': 0.001,
        'patch_shape': (51, 51),
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
        'image': 'data' + sep + 'DRIVE' + sep + 'images',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'data' + sep + 'DRIVE' + sep + 'patchnet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
    }
}

WIDE = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 32,
        'epochs': 30,
        'learning_rate': 0.001,
        'patch_shape': (51, 51),
        # 'patch_offset': (150, 150),
        # 'expand_patch_by': (184, 184),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': 'PATCHNET-WIDE.chk.tar',
        'log_frequency': 500,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'AV-WIDE' + sep + 'images',
        'mask': 'data' + sep + 'AV-WIDE' + sep + 'mask',
        'truth': 'data' + sep + 'AV-WIDE' + sep + 'manual',
        'logs': 'data' + sep + 'AV-WIDE' + sep + 'patchnet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None
    }
}
