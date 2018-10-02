import os

sep = os.sep

DRIVE = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 8,
        'epochs': 200,
        'learning_rate': 0.0001,
        'patch_shape': (32, 32),
        # 'patch_offset': (14, 14),
        'expand_patch_by': (16, 16),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': 'THRNET-DRIVE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE_UNET_MAP' + sep + 'images',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'data' + sep + 'DRIVE_UNET_MAP' + sep + 'thrnet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
    }
}

WIDE0 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 8,
        'epochs': 200,
        'learning_rate': 0.0001,
        'patch_shape': (32, 32),
        # 'patch_offset': (14, 14),
        'expand_patch_by': (16, 16),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': '0THRNET-WIDE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'AV-WIDE' + sep + 'images',
        'mask': 'data' + sep + 'AV-WIDE' + sep + 'mask',
        'truth': 'data' + sep + 'AV-WIDE' + sep + 'manual',
        'logs': 'data' + sep + 'AV-WIDE' + sep + 'unet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None
    }
}
WIDE5 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 8,
        'epochs': 200,
        'learning_rate': 0.0001,
        'patch_shape': (32, 32),
        # 'patch_offset': (14, 14),
        'expand_patch_by': (16, 16),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': '5THRNET-WIDE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'AV-WIDE' + sep + 'images',
        'mask': 'data' + sep + 'AV-WIDE' + sep + 'mask',
        'truth': 'data' + sep + 'AV-WIDE' + sep + 'manual',
        'logs': 'data' + sep + 'AV-WIDE' + sep + 'unet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None
    }
}
WIDE10 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 8,
        'epochs': 200,
        'learning_rate': 0.0001,
        'patch_shape': (32, 32),
        # 'patch_offset': (14, 14),
        'expand_patch_by': (16, 16),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': '10THRNET-WIDE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'AV-WIDE' + sep + 'images',
        'mask': 'data' + sep + 'AV-WIDE' + sep + 'mask',
        'truth': 'data' + sep + 'AV-WIDE' + sep + 'manual',
        'logs': 'data' + sep + 'AV-WIDE' + sep + 'unet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None
    }
}
WIDE15 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 8,
        'epochs': 200,
        'learning_rate': 0.0001,
        'patch_shape': (32, 32),
        # 'patch_offset': (14, 14),
        'expand_patch_by': (16, 16),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': '15THRNET-WIDE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'AV-WIDE' + sep + 'images',
        'mask': 'data' + sep + 'AV-WIDE' + sep + 'mask',
        'truth': 'data' + sep + 'AV-WIDE' + sep + 'manual',
        'logs': 'data' + sep + 'AV-WIDE' + sep + 'unet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None
    }
}
WIDE20 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 8,
        'epochs': 200,
        'learning_rate': 0.0001,
        'patch_shape': (32, 32),
        # 'patch_offset': (14, 14),
        'expand_patch_by': (16, 16),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': '20THRNET-WIDE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'AV-WIDE' + sep + 'images',
        'mask': 'data' + sep + 'AV-WIDE' + sep + 'mask',
        'truth': 'data' + sep + 'AV-WIDE' + sep + 'manual',
        'logs': 'data' + sep + 'AV-WIDE' + sep + 'unet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None
    }
}
WIDE25 = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 8,
        'epochs': 200,
        'learning_rate': 0.0001,
        'patch_shape': (32, 32),
        # 'patch_offset': (14, 14),
        'expand_patch_by': (16, 16),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': '25THRNET-WIDE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'AV-WIDE' + sep + 'images',
        'mask': 'data' + sep + 'AV-WIDE' + sep + 'mask',
        'truth': 'data' + sep + 'AV-WIDE' + sep + 'manual',
        'logs': 'data' + sep + 'AV-WIDE' + sep + 'unet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
        'mask_getter': None
    }
}