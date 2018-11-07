import os

sep = os.sep

DRIVE = {
    'Params': {
        'num_channels': 2,
        'num_classes': 2,
        'batch_size': 16,
        'epochs': 10,
        'learning_rate': 0.001,
        'patch_shape': (21, 21),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'log_frequency': 500,
        'validation_frequency': 1,
        'mode': 'test',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE' + sep + 'images',
        'image_unet': 'data' + sep + 'DRIVE' + sep + 'unet_del_logs',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'data' + sep + 'DRIVE' + sep + 'patchnet__logs',
        'splits_json': 'data' + sep + 'DRIVE' + sep + 'unet_splits'
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

STARE = {
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
        'checkpoint_file': 'PATCHNET-STARE.chk.tar',
        'log_frequency': 500,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'STARE' + sep + 'stare-images',
        'truth': 'data' + sep + 'STARE' + sep + 'labels-ah',
        'logs': 'data' + sep + 'STARE' + sep + 'patchnet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
        'mask_getter': None
    }
}

VEVIO = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 4,
        'epochs': 30,
        'learning_rate': 0.001,
        'patch_shape': (51, 51),
        # 'patch_offset': (150, 150),
        # 'expand_patch_by': (184, 184),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': 'PATCHNET-VEVIO.chk.tar',
        'log_frequency': 500,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'VEVIO' + sep + 'mosaics',
        'mask': 'data' + sep + 'VEVIO' + sep + 'mosaics_masks',
        'truth': 'data' + sep + 'VEVIO' + sep + 'mosaics_manual_01_bw',
        'logs': 'data' + sep + 'VEVIO' + sep + 'patchnet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.png',
        'mask_getter': lambda file_name: 'mask_' + file_name
    }
}
