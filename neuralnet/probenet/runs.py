import os
sep = os.sep

DEPTH_MAP = {
    'Params': {
        'num_channels': 1,
        'num_classes': 1,
        'batch_size': 4,
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
        'mode': 'train',
        'probe_mode': 'depth',
        'parallel_trained': False,
    },
    'Dirs': {
        'image': 'data' + sep + 'UNREAL' + sep + 'Grey',
        'truth': 'data' + sep + 'UNREAL' + sep + 'Depth',
        'logs': 'LOGS_2019' + sep + 'UNREAL' + sep + 'depth_map',
        'splits_json': 'data' + sep + 'UNREAL' + sep + 'splits-light'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name
    }
}