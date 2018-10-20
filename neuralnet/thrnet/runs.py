import os

sep = os.sep

DRIVE = {
    'Params': {
        'num_channels': 2,
        'num_classes': 1,
        'batch_size': 16,
        'epochs': 20,
        'learning_rate': 0.001,
        'patch_shape': (32, 32),
        'expand_patch_by': (0, 0),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE_MAP' + sep + 'images',
        'image_orig': 'data' + sep + 'DRIVE' + sep + 'images',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'data' + sep + 'DRIVE_MAP' + sep + 'thrnet_2_logs',
        'splits_json': 'data' + sep + 'DRIVE_MAP' + sep + 'thrnet_splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
    }
}

WIDE = {
    'Params': {
        'num_channels': 1,
        'num_classes': 1,
        'batch_size': 12,
        'epochs': 40,
        'learning_rate': 0.001,
        'patch_shape': (32, 32),
        'expand_patch_by': (16, 16),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'AV-WIDE_MAP' + sep + 'images',
        'mask': 'data' + sep + 'AV-WIDE' + sep + 'mask',
        'truth': 'data' + sep + 'AV-WIDE' + sep + 'manual',
        'logs': 'data' + sep + 'AV-WIDE_MAP' + sep + 'thrnet_logs',
        'splits_json': 'data' + sep + 'AV-WIDE_MAP' + sep + 'thrnet_splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None
    }
}

STARE = {
    'Params': {
        'num_channels': 1,
        'num_classes': 1,
        'batch_size': 12,
        'epochs': 40,
        'learning_rate': 0.001,
        'patch_shape': (32, 32),
        'expand_patch_by': (16, 16),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'STARE_MAP' + sep + 'images',
        'truth': 'data' + sep + 'STARE' + sep + 'labels-ah',
        'logs': 'data' + sep + 'STARE_MAP' + sep + 'thrnet_logs',
        'splits_json': 'data' + sep + 'STARE_MAP' + sep + 'thrnet_splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
        'mask_getter': None
    }
}


VEVIO = {
    'Params': {
        'num_channels': 2,
        'num_classes': 1,
        'batch_size': 12,
        'epochs': 40,
        'learning_rate': 0.001,
        'patch_shape': (32, 32),
        'expand_patch_by': (0, 0),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'VEVIO_MAP' + sep + 'images',
        'mask': 'data' + sep + 'VEVIO' + sep + 'mosaics_masks',
        'truth': 'data' + sep + 'VEVIO' + sep + 'mosaics_manual_01_bw',
        'logs': 'data' + sep + 'VEVIO_MAP' + sep + 'thrnet_logs',
        'splits_json': 'data' + sep + 'VEVIO_MAP' + sep + 'thrnet_splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.png',
        'mask_getter': lambda file_name: 'mask_' + file_name
    }
}