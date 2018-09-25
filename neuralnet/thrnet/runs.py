import os

sep = os.sep

DRIVE32a = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 8,
        'epochs': 200,
        'learning_rate': 0.001,
        'patch_shape': (32, 32),
        # 'patch_offset': (14, 14),
        'expand_patch_by': (0, 0),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': 'THRNET32a-DRIVE.chk.tar',
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


DRIVE32b = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 16,
        'epochs': 200,
        'learning_rate': 0.001,
        'patch_shape': (32, 32),
        # 'patch_offset': (14, 14),
        'expand_patch_by': (0, 0),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': 'THRNET32b-DRIVE.chk.tar',
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


DRIVE32c = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 10,
        'epochs': 200,
        'learning_rate': 0.001,
        'patch_shape': (32, 32),
        # 'patch_offset': (14, 14),
        'expand_patch_by': (0, 0),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': 'THRNET32c-DRIVE.chk.tar',
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






















DRIVE16 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 10,
        'epochs': 200,
        'learning_rate': 0.001,
        'patch_shape': (16, 16),
        # 'patch_offset': (14, 14),
        'expand_patch_by': (16, 16),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': 'THRNET16-DRIVE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'test',
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