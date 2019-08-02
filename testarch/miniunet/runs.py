import os

sep = os.sep
import testarch.unet.runs as R

################ HYPER PARAMETERS FOR MINI UNET ####################
Params = {
    'num_channels': 2,
    'num_classes': 2,
    'batch_size': 4,
    'epochs': 100,
    'learning_rate': 0.001,
    'patch_shape': (100, 100),
    'expand_patch_by': (40, 40),
    'use_gpu': True,
    'distribute': True,
    'shuffle': True,
    'log_frequency': 20,
    'validation_frequency': 1,
    'mode': 'train',
    'parallel_trained': False
}
######################################################################

# -----------------------------DRIVE----------------------------------------------
DRIVE_1_100_1 = R.DRIVE_1_100_1.copy()
DRIVE_1_100_1['Params'] = Params
DRIVE_1_100_1['Dirs']['image_unet'] = R.DRIVE_1_100_1['Dirs']['logs']
DRIVE_1_100_1['Dirs']['logs'] = 'logs' + sep + 'DRIVE' + sep + 'MINI_UNET_1_100_1'

DRIVE_1_1 = R.DRIVE_1_1.copy()
DRIVE_1_1['Params'] = Params
DRIVE_1_1['Dirs']['image_unet'] = R.DRIVE_1_1['Dirs']['logs']
DRIVE_1_1['Dirs']['logs'] = 'logs' + sep + 'DRIVE' + sep + 'MINI_UNET_1__1'

DRIVE_WEIGHTED = R.DRIVE_WEIGHTED.copy()
DRIVE_WEIGHTED['Params'] = Params
DRIVE_WEIGHTED['Dirs']['image_unet'] = R.DRIVE_WEIGHTED['Dirs']['logs']
DRIVE_WEIGHTED['Dirs']['logs'] = 'logs' + sep + 'DRIVE' + sep + 'MINI_UNET_WEIGHTED'
# ----------------------------------------------------------------------------------

# -----------------------------AV_WIDE-----------------------------------------------
WIDE_1_100_1 = R.WIDE_1_100_1.copy()
WIDE_1_100_1['Params'] = Params
WIDE_1_100_1['Dirs']['image_unet'] = R.WIDE_1_100_1['Dirs']['logs']
WIDE_1_100_1['Dirs']['logs'] = 'logs' + sep + 'AV_WIDE' + sep + 'MINI_UNET_1_100_1'

WIDE_1_1 = R.WIDE_1_1.copy()
WIDE_1_1['Params'] = Params
WIDE_1_1['Dirs']['image_unet'] = R.WIDE_1_1['Dirs']['logs']
WIDE_1_1['Dirs']['logs'] = 'logs' + sep + 'AV_WIDE' + sep + 'MINI_UNET_1__1'

WIDE_WEIGHTED = R.WIDE_WEIGHTED.copy()
WIDE_WEIGHTED['Params'] = Params
WIDE_WEIGHTED['Dirs']['image_unet'] = R.WIDE_WEIGHTED['Dirs']['logs']
WIDE_WEIGHTED['Dirs']['logs'] = 'logs' + sep + 'AV_WIDE' + sep + 'MINI_UNET_WEIGHTED'
# -------------------------------------------------------------------------------------

# -----------------------------STARE---------------------------------------------------
STARE_1_100_1 = R.STARE_1_100_1.copy()
STARE_1_100_1['Params'] = Params
STARE_1_100_1['Dirs']['image_unet'] = R.STARE_1_100_1['Dirs']['logs']
STARE_1_100_1['Dirs']['logs'] = 'logs' + sep + 'STARE' + sep + 'MINI_UNET_1_100_1'

STARE_1_1 = R.STARE_1_1.copy()
STARE_1_1['Params'] = Params
STARE_1_1['Dirs']['image_unet'] = R.STARE_1_1['Dirs']['logs']
STARE_1_1['Dirs']['logs'] = 'logs' + sep + 'STARE' + sep + 'MINI_UNET_1__1'

STARE_WEIGHTED = R.STARE_WEIGHTED.copy()
STARE_WEIGHTED['Params'] = Params
STARE_WEIGHTED['Dirs']['image_unet'] = R.STARE_WEIGHTED['Dirs']['logs']
STARE_WEIGHTED['Dirs']['logs'] = 'logs' + sep + 'STARE' + sep + 'MINI_UNET_WEIGHTED'
# ----------------------------------------------------------------------------------------


# -----------------------------CHASEDB----------------------------------------------------
CHASEDB_1_100_1 = R.CHASEDB_1_100_1.copy()
CHASEDB_1_100_1['Params'] = Params
CHASEDB_1_100_1['Dirs']['image_unet'] = R.CHASEDB_1_100_1['Dirs']['logs']
CHASEDB_1_100_1['Dirs']['logs'] = 'logs' + sep + 'CHASEDB' + sep + 'MINI_UNET_1_100_1'

CHASEDB_1_1 = R.CHASEDB_1_1.copy()
CHASEDB_1_1['Params'] = Params
CHASEDB_1_1['Dirs']['image_unet'] = R.CHASEDB_1_1['Dirs']['logs']
CHASEDB_1_1['Dirs']['logs'] = 'logs' + sep + 'CHASEDB' + sep + 'MINI_UNET_1__1'

CHASEDB_WEIGHTED = R.CHASEDB_WEIGHTED.copy()
CHASEDB_WEIGHTED['Params'] = Params
CHASEDB_WEIGHTED['Dirs']['image_unet'] = R.CHASEDB_WEIGHTED['Dirs']['logs']
CHASEDB_WEIGHTED['Dirs']['logs'] = 'logs' + sep + 'CHASEDB' + sep + 'MINI_UNET_WEIGHTED'
# -----------------------------------------------------------------------------------------


# -----------------------------VEVIO-MOSAICS---------------------------------------------------------
VEVIO_MOSAICS_1_100_1 = R.VEVIO_MOSAICS_1_100_1.copy()
VEVIO_MOSAICS_1_100_1['Params'] = Params
VEVIO_MOSAICS_1_100_1['Dirs']['image_unet'] = R.VEVIO_MOSAICS_1_100_1['Dirs']['logs']
VEVIO_MOSAICS_1_100_1['Dirs']['logs'] = 'logs' + sep + 'VEVIO_MOSAICS' + sep + 'MINI_UNET_1_100_1'

VEVIO_MOSAICS_1_1 = R.VEVIO_MOSAICS_1_1.copy()
VEVIO_MOSAICS_1_1['Params'] = Params
VEVIO_MOSAICS_1_1['Dirs']['image_unet'] = R.VEVIO_MOSAICS_1_1['Dirs']['logs']
VEVIO_MOSAICS_1_1['Dirs']['logs'] = 'logs' + sep + 'VEVIO_MOSAICS' + sep + 'MINI_UNET_1__1'

VEVIO_MOSAICS_WEIGHTED = R.VEVIO_MOSAICS_WEIGHTED.copy()
VEVIO_MOSAICS_WEIGHTED['Params'] = Params
VEVIO_MOSAICS_WEIGHTED['Dirs']['image_unet'] = R.VEVIO_MOSAICS_WEIGHTED['Dirs']['logs']
VEVIO_MOSAICS_WEIGHTED['Dirs']['logs'] = 'logs' + sep + 'VEVIO_MOSAICS' + sep + 'MINI_UNET_WEIGHTED'
# -------------------------------------------------------------------------------------------------------

# -----------------------------VEVIO_FRAMES--------------------------------------------------------------
VEVIO_FRAMES_1_100_1 = R.VEVIO_FRAMES_1_100_1.copy()
VEVIO_FRAMES_1_100_1['Params'] = Params
VEVIO_FRAMES_1_100_1['Dirs']['image_unet'] = R.VEVIO_FRAMES_1_100_1['Dirs']['logs']
VEVIO_FRAMES_1_100_1['Dirs']['logs'] = 'logs' + sep + 'VEVIO_FRAMES' + sep + 'MINI_UNET_1_100_1'

VEVIO_FRAMES_1_1 = R.VEVIO_FRAMES_1_1.copy()
VEVIO_FRAMES_1_1['Params'] = Params
VEVIO_FRAMES_1_1['Dirs']['image_unet'] = R.VEVIO_FRAMES_1_1['Dirs']['logs']
VEVIO_FRAMES_1_1['Dirs']['logs'] = 'logs' + sep + 'VEVIO_FRAMES' + sep + 'MINI_UNET_1__1'

VEVIO_FRAMES_WEIGHTED = R.VEVIO_FRAMES_WEIGHTED.copy()
VEVIO_FRAMES_WEIGHTED['Params'] = Params
VEVIO_FRAMES_WEIGHTED['Dirs']['image_unet'] = R.VEVIO_FRAMES_WEIGHTED['Dirs']['logs']
VEVIO_FRAMES_WEIGHTED['Dirs']['logs'] = 'logs' + sep + 'VEVIO_FRAMES' + sep + 'MINI_UNET_WEIGHTED'
# ---------------------------------------------------------------------------------------------------------
