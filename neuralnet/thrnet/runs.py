import os

sep = os.sep

DRIVE16 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 16,
        'epochs': 200,
        'learning_rate': 0.001,
        'patch_shape': (16, 16),
        'patch_offset': (14, 14),
        'expand_patch_by': (16, 16),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'checkpoint_file': 'THRNET16-DRIVE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE_UNET_MAP' + sep + 'images',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'data' + sep + 'DRIVE_UNET_MAP' + sep + 'thrnet16_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
    }
}

DRIVE32 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 1,
        'batch_size': 16,
        'epochs': 200,
        'learning_rate': 0.001,
        'patch_shape': (32, 32),
        'patch_offset': (20, 20),
        'expand_patch_by': (0, 0),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'checkpoint_file': 'THRNET32-DRIVE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE_UNET_MAP' + sep + 'images',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'data' + sep + 'DRIVE_UNET_MAP' + sep + 'thrnet32_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
    }
}

DRIVE64 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 1,
        'batch_size': 16,
        'epochs': 200,
        'learning_rate': 0.001,
        'patch_shape': (64, 64),
        'patch_offset': (30, 30),
        'expand_patch_by': (64, 64),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'checkpoint_file': 'THRNET64-DRIVE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 2,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE_UNET_MAP' + sep + 'images',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'data' + sep + 'DRIVE_UNET_MAP' + sep + 'thrnet64_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
    }
}
