import copy
import os

import numpy as np

sep = os.sep

####################### GLOBAL PARAMETERS ##################################################
############################################################################################
Params = {
    'num_channels': 1,
    'num_classes': 2,
    'batch_size': 4,
    'epochs': 351,
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
}

# dparm_1_100_1 = lambda x: (1 + np.random.beta(.5, .5, 2) * 100).astype(int)
dparm_1_100_1 = lambda x: np.random.choice(np.arange(1, 101, 1), 2)
# dparm_1_1 = lambda x: [1, 1]
# d_parm_weighted = lambda x: [x['Params']['cls_weights'][0], x['Params']['cls_weights'][1]]
##############################################################################################

DRIVE = {
    'Params': Params,
    'Dirs': {
        'image': 'data' + sep + 'DRIVE' + sep + 'images',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'splits_json': 'data' + sep + 'DRIVE' + sep + 'splits'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
    }
}

DRIVE_1_100_1 = copy.deepcopy(DRIVE)
DRIVE_1_100_1['Dirs']['logs'] = 'logs' + sep + 'DRIVE' + sep + 'UNET_1_100_1'
DRIVE_1_100_1['Funcs']['dparm'] = dparm_1_100_1
#
# DRIVE_1_1 = copy.deepcopy(DRIVE)
# DRIVE_1_1['Dirs']['logs'] = 'logs' + sep + 'DRIVE' + sep + 'UNET_1_1'
# DRIVE_1_1['Funcs']['dparm'] = dparm_1_1
#
# DRIVE_WEIGHTED = copy.deepcopy(DRIVE)
# DRIVE_WEIGHTED['Dirs']['logs'] = 'logs' + sep + 'DRIVE' + sep + 'UNET_WEIGHTED'
# DRIVE_WEIGHTED['Funcs']['dparm'] = d_parm_weighted
# # --------------------------------------------------------------------------------------------
#
# WIDE = {
#     'Params': Params,
#     'Dirs': {
#         'image': 'data' + sep + 'AV-WIDE' + sep + 'images',
#         'truth': 'data' + sep + 'AV-WIDE' + sep + 'manual',
#         'splits_json': 'data' + sep + 'AV-WIDE' + sep + 'splits'
#     },
#
#     'Funcs': {
#         'truth_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
#         'mask_getter': None
#     }
# }
#
# WIDE_1_100_1 = copy.deepcopy(WIDE)
# WIDE_1_100_1['Dirs']['logs'] = 'logs' + sep + 'AV_WIDE' + sep + 'UNET_1_100_1'
# WIDE_1_100_1['Funcs']['dparm'] = dparm_1_100_1
#
# WIDE_1_1 = copy.deepcopy(WIDE)
# WIDE_1_1['Dirs']['logs'] = 'logs' + sep + 'AV_WIDE' + sep + 'UNET_1_1'
# WIDE_1_1['Funcs']['dparm'] = dparm_1_1
#
# WIDE_WEIGHTED = copy.deepcopy(WIDE)
# WIDE_WEIGHTED['Dirs']['logs'] = 'logs' + sep + 'AV_WIDE' + sep + 'UNET_WEIGHTED'
# WIDE_WEIGHTED['Funcs']['dparm'] = d_parm_weighted
# # ---------------------------------------------------------------------------------------------
#
# STARE = {
#     'Params': Params,
#     'Dirs': {
#         'image': 'data' + sep + 'STARE' + sep + 'stare-images',
#         'truth': 'data' + sep + 'STARE' + sep + 'labels-ah',
#         'splits_json': 'data' + sep + 'STARE' + sep + 'splits'
#     },
#
#     'Funcs': {
#         'truth_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
#         'mask_getter': None
#     }
# }
#
# STARE_1_100_1 = copy.deepcopy(STARE)
# STARE_1_100_1['Dirs']['logs'] = 'logs' + sep + 'STARE' + sep + 'UNET_1_100_1'
# STARE_1_100_1['Funcs']['dparm'] = dparm_1_100_1
#
# STARE_1_1 = copy.deepcopy(STARE)
# STARE_1_1['Dirs']['logs'] = 'logs' + sep + 'STARE' + sep + 'UNET_1_1'
# STARE_1_1['Funcs']['dparm'] = dparm_1_1
#
# STARE_WEIGHTED = copy.deepcopy(STARE)
# STARE_WEIGHTED['Dirs']['logs'] = 'logs' + sep + 'STARE' + sep + 'UNET_WEIGHTED'
# STARE_WEIGHTED['Funcs']['dparm'] = d_parm_weighted
# # ------------------------------------------------------------------------------------------------
#
# CHASEDB = {
#     'Params': Params,
#     'Dirs': {
#         'image': 'data' + sep + 'CHASEDB' + sep + 'images',
#         'truth': 'data' + sep + 'CHASEDB' + sep + 'manual',
#         'splits_json': 'data' + sep + 'CHASEDB' + sep + 'splits'
#     },
#
#     'Funcs': {
#         'truth_getter': lambda file_name: file_name.split('.')[0] + '_1stHO.png',
#         'mask_getter': None
#     }
# }
#
# CHASEDB_1_100_1 = copy.deepcopy(CHASEDB)
# CHASEDB_1_100_1['Dirs']['logs'] = 'logs' + sep + 'CHASEDB' + sep + 'UNET_1_100_1'
# CHASEDB_1_100_1['Funcs']['dparm'] = dparm_1_100_1
#
# CHASEDB_1_1 = copy.deepcopy(CHASEDB)
# CHASEDB_1_1['Dirs']['logs'] = 'logs' + sep + 'CHASEDB' + sep + 'UNET_1_1'
# CHASEDB_1_1['Funcs']['dparm'] = dparm_1_1
#
# CHASEDB_WEIGHTED = copy.deepcopy(CHASEDB)
# CHASEDB_WEIGHTED['Dirs']['logs'] = 'logs' + sep + 'CHASEDB' + sep + 'UNET_WEIGHTED'
# CHASEDB_WEIGHTED['Funcs']['dparm'] = d_parm_weighted
# # -------------------------------------------------------------------------------------------------
#
# VEVIO_MOSAICS = {
#     'Params': Params,
#     'Dirs': {
#         'image': 'data' + sep + 'VEVIO' + sep + 'mosaics',
#         'mask': 'data' + sep + 'VEVIO' + sep + 'mosaics_masks',
#         'truth': 'data' + sep + 'VEVIO' + sep + 'mosaics_manual_01_bw',
#         'splits_json': 'data' + sep + 'VEVIO' + sep + 'splits_mosaics'
#     },
#
#     'Funcs': {
#         'truth_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.' + file_name.split('.')[1],
#         'mask_getter': lambda file_name: 'mask_' + file_name
#     }
# }
#
# VEVIO_MOSAICS_1_100_1 = copy.deepcopy(VEVIO_MOSAICS)
# VEVIO_MOSAICS_1_100_1['Dirs']['logs'] = 'logs' + sep + 'VEVIO_MOSAICS' + sep + 'UNET_1_100_1'
# VEVIO_MOSAICS_1_100_1['Funcs']['dparm'] = dparm_1_100_1
#
# VEVIO_MOSAICS_1_1 = copy.deepcopy(VEVIO_MOSAICS)
# VEVIO_MOSAICS_1_1['Dirs']['logs'] = 'logs' + sep + 'VEVIO_MOSAICS' + sep + 'UNET_1_1'
# VEVIO_MOSAICS_1_1['Funcs']['dparm'] = dparm_1_1
#
# VEVIO_MOSAICS_WEIGHTED = copy.deepcopy(VEVIO_MOSAICS)
# VEVIO_MOSAICS_WEIGHTED['Dirs']['logs'] = 'logs' + sep + 'VEVIO_MOSAICS' + sep + 'UNET_WEIGHTED'
# VEVIO_MOSAICS_WEIGHTED['Funcs']['dparm'] = d_parm_weighted
# # ---------------------------------------------------------------------------------------------------------
#
# VEVIO_FRAMES = {
#     'Params': Params,
#     'Dirs': {
#         'image': 'data' + sep + 'VEVIO' + sep + 'frames',
#         'mask': 'data' + sep + 'VEVIO' + sep + 'frames_masks',
#         'truth': 'data' + sep + 'VEVIO' + sep + 'frames_manual_01_bw',
#         'splits_json': 'data' + sep + 'VEVIO' + sep + 'splits_frames'
#     },
#
#     'Funcs': {
#         'truth_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.' + file_name.split('.')[1],
#         'mask_getter': lambda file_name: 'mask_' + file_name
#     }
# }
#
# VEVIO_FRAMES_1_100_1 = copy.deepcopy(VEVIO_FRAMES)
# VEVIO_FRAMES_1_100_1['Dirs']['logs'] = 'logs' + sep + 'VEVIO_FRAMES' + sep + 'UNET_1_100_1'
# VEVIO_FRAMES_1_100_1['Funcs']['dparm'] = dparm_1_100_1
#
# VEVIO_FRAMES_1_1 = copy.deepcopy(VEVIO_FRAMES)
# VEVIO_FRAMES_1_1['Dirs']['logs'] = 'logs' + sep + 'VEVIO_FRAMES' + sep + 'UNET_1_1'
# VEVIO_FRAMES_1_1['Funcs']['dparm'] = dparm_1_1
#
# VEVIO_FRAMES_WEIGHTED = copy.deepcopy(VEVIO_FRAMES)
# VEVIO_FRAMES_WEIGHTED['Dirs']['logs'] = 'logs' + sep + 'VEVIO_FRAMES' + sep + 'UNET_WEIGHTED'
# VEVIO_FRAMES_WEIGHTED['Funcs']['dparm'] = d_parm_weighted
# # -------------------------------------------------------------------------------------------------------------
