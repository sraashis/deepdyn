import os

sep = os.sep

DRIVE = {

    'P': {
        'num_channels': 1,
        'num_classes': 1,
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'patch_shape': (21, 21),
        'use_gpu': True,
        'distribute': True,
        'shuffle': True,
        'checkpoint_file': 'THRNET-DRIVE.chk.tar',
        'mode': 'eval'
    },

    'D': {
        'train_img': 'data' + sep + 'DRIVE' + sep + 'thr_training' + sep + 'images',
        'train_mask': 'data' + sep + 'DRIVE' + sep + 'thr_training' + sep + 'mask',
        'train_manual': 'data' + sep + 'DRIVE' + sep + 'thr_training' + sep + '1st_manual',

        'val_img': 'data' + sep + 'DRIVE' + sep + 'thr_testing' + sep + 'validation_images',
        'val_mask': 'data' + sep + 'DRIVE' + sep + 'thr_testing' + sep + 'mask',
        'val_manual': 'data' + sep + 'DRIVE' + sep + 'thr_testing' + sep + '1st_manual',

        'test_img': 'data' + sep + 'DRIVE' + sep + 'thr_testing' + sep + 'images',
        'test_mask': 'data' + sep + 'DRIVE' + sep + 'thr_testing' + sep + 'mask',
        'test_manual': 'data' + sep + 'DRIVE' + sep + 'thr_testing' + sep + '1st_manual',

        'test_img_out': 'data' + sep + 'DRIVE' + sep + 'thr_testing' + sep + 'thrnet_out'},

    'F': {
        'train_gt_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'train_mask_getter': lambda file_name: file_name.split('_')[0] + '_test_mask.gif',
        'val_gt_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'val_mask_getter': lambda file_name: file_name.split('_')[0] + '_test_mask.gif',
        'test_gt_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'test_mask_getter': lambda file_name: file_name.split('_')[0] + '_test_mask.gif',

    }
}
