import os

sep = os.sep
import numpy as np

DRIVE = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 4,
        'epochs': 250,
        'learning_rate': 0.001,
        'patch_shape': (388, 388),
        'patch_offset': (150, 150),
        'expand_patch_by': (184, 184),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'log_frequency': 5,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False,
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE' + sep + 'images',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'LOGS_2019' + sep + 'DRIVE' + sep + 'UNET_1_1',
        'splits_json': 'data' + sep + 'DRIVE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif',
        'dparm': lambda p, r: [1, 1]
    }
}

DRIVE1 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 4,
        'epochs': 250,
        'learning_rate': 0.001,
        'patch_shape': (388, 388),
        'patch_offset': (150, 150),
        'expand_patch_by': (184, 184),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'log_frequency': 5,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE' + sep + 'images',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'LOGS_2019' + sep + 'DRIVE' + sep + 'UNET_1_100_1',
        'splits_json': 'data' + sep + 'DRIVE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif',
        'dparm': lambda p, r: np.random.choice(np.arange(1, 101, 1), 2)
    }
}

WIDE = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 4,
        'epochs': 250,
        'learning_rate': 0.001,
        'patch_shape': (388, 388),
        'patch_offset': (150, 150),
        'expand_patch_by': (184, 184),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'log_frequency': 5,
        'validation_frequency': 1,
        'mode': 'test',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'AV-WIDE' + sep + 'images',
        'mask': 'data' + sep + 'AV-WIDE' + sep + 'mask',
        'truth': 'data' + sep + 'AV-WIDE' + sep + 'manual',
        'logs': 'LOGS_2019' + sep + 'AV-WIDE' + sep + 'UNET_1_1',
        'splits_json': 'data' + sep + 'AV-WIDE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None,
        'dparm': lambda p, r: [1, 1]
    }
}

WIDE1 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 4,
        'epochs': 250,
        'learning_rate': 0.001,
        'patch_shape': (388, 388),
        'patch_offset': (150, 150),
        'expand_patch_by': (184, 184),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'log_frequency': 5,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'AV-WIDE' + sep + 'images',
        'mask': 'data' + sep + 'AV-WIDE' + sep + 'mask',
        'truth': 'data' + sep + 'AV-WIDE' + sep + 'manual',
        'logs': 'LOGS_2019' + sep + 'AV-WIDE' + sep + 'UNET_1_100_1',
        'splits_json': 'data' + sep + 'AV-WIDE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None,
        'dparm': lambda p, r: np.random.choice(np.arange(1, 101, 1), 2)
    }
}

STARE = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 4,
        'epochs': 200,
        'learning_rate': 0.001,
        'patch_shape': (388, 388),
        'patch_offset': (150, 150),
        'expand_patch_by': (184, 184),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'log_frequency': 5,
        'validation_frequency': 1,
        'mode': 'test',
        'parallel_trained': False

    },
    'Dirs': {
        'image': 'data' + sep + 'STARE' + sep + 'stare-images',
        'truth': 'data' + sep + 'STARE' + sep + 'labels-ah',
        'logs': 'LOGS_2019' + sep + 'STARE' + sep + 'UNET_1_1',
        'splits_json': 'data' + sep + 'STARE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
        'mask_getter': None,
        'dparm': lambda p, r: [1, 1]
    }
}

STARE1 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 4,
        'epochs': 250,
        'learning_rate': 0.001,
        'patch_shape': (388, 388),
        'patch_offset': (150, 150),
        'expand_patch_by': (184, 184),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'log_frequency': 5,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'STARE' + sep + 'stare-images',
        'truth': 'data' + sep + 'STARE' + sep + 'labels-ah',
        'logs': 'LOGS_2019' + sep + 'STARE' + sep + 'UNET_1_100_1',
        'splits_json': 'data' + sep + 'STARE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
        'mask_getter': None,
        'dparm': lambda p, r: np.random.choice(np.arange(1, 101, 1), 2)
    }
}

VEVIO = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 4,
        'epochs': 200,
        'learning_rate': 0.001,
        'patch_shape': (388, 388),
        'patch_offset': (150, 150),
        'expand_patch_by': (184, 184),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'log_frequency': 5,
        'validation_frequency': 1,
        'mode': 'test',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'VEVIO' + sep + 'mosaics',
        'mask': 'data' + sep + 'VEVIO' + sep + 'mosaics_masks',
        'truth': 'data' + sep + 'VEVIO' + sep + 'mosaics_manual_01_bw',
        'logs': 'LOGS_2019' + sep + 'VEVIO' + sep + 'UNET_1_1',
        'splits_json': 'data' + sep + 'VEVIO' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.png',
        'mask_getter': lambda file_name: 'mask_' + file_name,
        'dparm': lambda p, r: [1, 1]
    }
}

VEVIO1 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 4,
        'epochs': 200,
        'learning_rate': 0.001,
        'patch_shape': (388, 388),
        'patch_offset': (150, 150),
        'expand_patch_by': (184, 184),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'log_frequency': 5,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'VEVIO' + sep + 'mosaics',
        'mask': 'data' + sep + 'VEVIO' + sep + 'mosaics_masks',
        'truth': 'data' + sep + 'VEVIO' + sep + 'mosaics_manual_01_bw',
        'logs': 'LOGS_2019' + sep + 'VEVIO' + sep + 'UNET_1_100_1',
        'splits_json': 'data' + sep + 'VEVIO' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.png',
        'mask_getter': lambda file_name: 'mask_' + file_name,
        'dparm': lambda p, r: np.random.choice(np.arange(1, 101, 1), 2)
    }
}
