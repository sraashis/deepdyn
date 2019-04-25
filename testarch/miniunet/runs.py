import os

sep = os.sep

DRIVE = {
    'Params': {
        'num_channels': 2,
        'num_classes': 2,
        'batch_size': 4,
        'epochs': 100,
        'learning_rate': 0.001,
        'patch_shape': (100, 100),
        'expand_patch_by': (40, 40),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'log_frequency': 20,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE' + sep + 'images',
        'image_unet': 'logs' + sep + 'DRIVE' + sep + 'UNET',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'logs' + sep + 'DRIVE' + sep + 'MINI-UNET',
        'splits_json': 'data' + sep + 'DRIVE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
    }
}
#####################################################################################################
WIDE = {
    'Params': {
        'num_channels': 2,
        'num_classes': 2,
        'batch_size': 6,
        'epochs': 61,
        'learning_rate': 0.001,
        'patch_shape': (100, 100),
        'expand_patch_by': (40, 40),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'test',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'AV-WIDE' + sep + 'images',
        'image_unet': 'data' + sep + 'AV-WIDE' + sep + 'UNET_LOGS',
        'mask': 'data' + sep + 'AV-WIDE' + sep + 'mask',
        'truth': 'data' + sep + 'AV-WIDE' + sep + 'manual',
        'logs': 'data' + sep + 'AV-WIDE' + sep + 'MAPNET_LOGS',
        'splits_json': 'data' + sep + 'AV-WIDE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None
    }
}

STARE = {
    'Params': {
        'num_channels': 2,
        'num_classes': 2,
        'batch_size': 6,
        'epochs': 65,
        'learning_rate': 0.001,
        'patch_shape': (100, 100),
        'expand_patch_by': (40, 40),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'test',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'STARE' + sep + 'stare-images',
        'image_unet': 'data' + sep + 'STARE' + sep + 'UNET_LOGS',
        'truth': 'data' + sep + 'STARE' + sep + 'labels-ah',
        'logs': 'data' + sep + 'STARE' + sep + 'MAPNET_LOGS',
        'splits_json': 'data' + sep + 'STARE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
        'mask_getter': None
    }
}

VEVIO = {
    'Params': {
        'num_channels': 2,
        'num_classes': 2,
        'batch_size': 6,
        'epochs': 61,
        'learning_rate': 0.001,
        'patch_shape': (100, 100),
        'expand_patch_by': (40, 40),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'VEVIO' + sep + 'frames',
        'image_unet': 'LOGS_2019' + sep + 'VEVIO_FRAMES' + sep + 'UNET_WEIGHTED',
        'mask': 'data' + sep + 'VEVIO' + sep + 'frames_masks',
        'truth': 'data' + sep + 'VEVIO' + sep + 'frames_manual_01_bw',
        'logs': 'LOGS_2019' + sep + 'VEVIO_FRAMES' + sep + 'MAPNET_WEIGHTED',
        'splits_json': 'data' + sep + 'VEVIO' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.' + file_name.split('.')[1],
        'mask_getter': lambda file_name: 'mask_' + file_name,
    }
}

VEVIO1 = {
    'Params': {
        'num_channels': 2,
        'num_classes': 2,
        'batch_size': 6,
        'epochs': 61,
        'learning_rate': 0.001,
        'patch_shape': (100, 100),
        'expand_patch_by': (40, 40),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'VEVIO' + sep + 'frames',
        'image_unet': 'LOGS_2019' + sep + 'VEVIO_FRAMES' + sep + 'UNET_1_100_1',
        'mask': 'data' + sep + 'VEVIO' + sep + 'frames_masks',
        'truth': 'data' + sep + 'VEVIO' + sep + 'frames_manual_01_bw',
        'logs': 'LOGS_2019' + sep + 'VEVIO_FRAMES' + sep + 'MAPNET_1_100_1',
        'splits_json': 'data' + sep + 'VEVIO' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.' + file_name.split('.')[1],
        'mask_getter': lambda file_name: 'mask_' + file_name,
    }
}
