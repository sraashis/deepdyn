import os
import numpy as np
sep = os.sep

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
        'logs': 'LOGS_2019' + sep + 'DRIVE' + sep + 'UNET_1_100_1',
        'splits_json': 'data' + sep + 'DRIVE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif',
        'dparm': lambda x: np.random.choice(np.arange(1, 101, 1), 2)
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
        'logs': 'LOGS_2019' + sep + 'DRIVE' + sep + 'UNET_1_1',
        'splits_json': 'data' + sep + 'DRIVE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif',
        'dparm': lambda x: [1, 1]
    }
}
DRIVE2 = {
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
        'logs': 'LOGS_2019' + sep + 'DRIVE' + sep + 'UNET_WEIGHTED',
        'splits_json': 'data' + sep + 'DRIVE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif',
        'dparm': lambda x: [x['Params']['cls_weights'][0], x['Params']['cls_weights'][1]]
    }
}
DRIVE3 = {
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
        'logs': 'LOGS_2019' + sep + 'DRIVE' + sep + 'UNET_1_10_1',
        'splits_json': 'data' + sep + 'DRIVE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif',
        'dparm': lambda x: np.random.choice(np.arange(1, 10, 1), 2)
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
        'dparm': lambda x: np.random.choice(np.arange(1, 101, 1), 2)
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
        'logs': 'LOGS_2019' + sep + 'AV-WIDE' + sep + 'UNET_1_1',
        'splits_json': 'data' + sep + 'AV-WIDE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None,
        'dparm': lambda x: [1, 1]
    }
}
WIDE2 = {
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
        'logs': 'LOGS_2019' + sep + 'AV-WIDE' + sep + 'UNET_WEIGHTED',
        'splits_json': 'data' + sep + 'AV-WIDE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None,
        'dparm': lambda x: [x['Params']['cls_weights'][0], x['Params']['cls_weights'][1]]
    }
}
WIDE3 = {
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
        'logs': 'LOGS_2019' + sep + 'AV-WIDE' + sep + 'UNET_1_10_1',
        'splits_json': 'data' + sep + 'AV-WIDE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None,
        'dparm': lambda x: np.random.choice(np.arange(1, 10, 1), 2)
    }
}

STARE = {
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
        'dparm': lambda x: np.random.choice(np.arange(1, 101, 1), 2)
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
        'logs': 'LOGS_2019' + sep + 'STARE' + sep + 'UNET_1_1',
        'splits_json': 'data' + sep + 'STARE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
        'mask_getter': None,
        'dparm': lambda x: [1, 1]
    }
}
STARE2 = {
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
        'logs': 'LOGS_2019' + sep + 'STARE' + sep + 'UNET_WEIGHTED',
        'splits_json': 'data' + sep + 'STARE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
        'mask_getter': None,
        'dparm': lambda x: [x['Params']['cls_weights'][0], x['Params']['cls_weights'][1]]
    }
}
STARE3 = {
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
        'logs': 'LOGS_2019' + sep + 'STARE' + sep + 'UNET_1_10_1',
        'splits_json': 'data' + sep + 'STARE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
        'mask_getter': None,
        'dparm': lambda x: np.random.choice(np.arange(1, 10, 1), 2)
    }
}

VEVIO = {
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
        'image': 'data' + sep + 'VEVIO' + sep + 'mosaics',
        'mask': 'data' + sep + 'VEVIO' + sep + 'mosaics_masks',
        'truth': 'data' + sep + 'VEVIO' + sep + 'mosaics_manual_01_bw',
        'logs': 'LOGS_2019' + sep + 'VEVIO' + sep + 'UNET_1_100_1',
        'splits_json': 'data' + sep + 'VEVIO' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.png',
        'mask_getter': lambda file_name: 'mask_' + file_name,
        'dparm': lambda x: np.random.choice(np.arange(1, 101, 1), 2)
    }
}
VEVIO1 = {
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
        'image': 'data' + sep + 'VEVIO' + sep + 'mosaics',
        'mask': 'data' + sep + 'VEVIO' + sep + 'mosaics_masks',
        'truth': 'data' + sep + 'VEVIO' + sep + 'mosaics_manual_01_bw',
        'logs': 'LOGS_2019' + sep + 'VEVIO' + sep + 'UNET_1_1',
        'splits_json': 'data' + sep + 'VEVIO' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.png',
        'mask_getter': lambda file_name: 'mask_' + file_name,
        'dparm': lambda x: [1, 1]
    }
}
VEVIO2 = {
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
        'image': 'data' + sep + 'VEVIO' + sep + 'mosaics',
        'mask': 'data' + sep + 'VEVIO' + sep + 'mosaics_masks',
        'truth': 'data' + sep + 'VEVIO' + sep + 'mosaics_manual_01_bw',
        'logs': 'LOGS_2019' + sep + 'VEVIO' + sep + 'UNET_WEIGHTED',
        'splits_json': 'data' + sep + 'VEVIO' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.png',
        'mask_getter': lambda file_name: 'mask_' + file_name,
        'dparm': lambda x: [x['Params']['cls_weights'][0], x['Params']['cls_weights'][1]]
    }
}
VEVIO3 = {
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
        'image': 'data' + sep + 'VEVIO' + sep + 'mosaics',
        'mask': 'data' + sep + 'VEVIO' + sep + 'mosaics_masks',
        'truth': 'data' + sep + 'VEVIO' + sep + 'mosaics_manual_01_bw',
        'logs': 'LOGS_2019' + sep + 'VEVIO' + sep + 'UNET_1_10_1',
        'splits_json': 'data' + sep + 'VEVIO' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.png',
        'mask_getter': lambda file_name: 'mask_' + file_name,
        'dparm': lambda x: np.random.choice(np.arange(1, 10, 1), 2)
    }
}
