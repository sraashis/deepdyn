import os
sep = os.sep

GREY_HIGH = {
    'Params': {
        'num_channels': 3,
        'num_classes': 3,
        'batch_size': 1,
        'epochs': 10,
        'learning_rate': 0.001,
        'patch_shape': (388, 388),
        'patch_offset': (150, 150),
        'expand_patch_by': (184, 184),
        'use_gpu': True,
        'distribute': False,
        'shuffle': True,
        'log_frequency': 5,
        'validation_frequency': 1,
        'mode': 'test',
        'parallel_trained': False,
    },
    'Dirs': {
        'image': 'data' + sep + 'Day' + sep + 'Grey',
        'truth': 'data' + sep + 'Day' + sep + 'Depth',
        'logs': 'LOGS_2019' + sep + 'Day' + sep + 'UNET_NORMAL_DAY',
        'splits_json': 'data' + sep + 'Day' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name
    }
}