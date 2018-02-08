##############################################################################
# IMPORTS, FLAGS, AND FOLDERS
##############################################################################
import scipy
from scipy import ndimage
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Global parameters
plt.rcParams['figure.figsize'] = [14,14]

# Flags
Flags = {}
Flags['readImages'] =   True
Flags['fixMasks'] =     True
Flags['useTrain'] =     False #True

# Folders
Dirs = {}
if Flags['useTrain']:
    folder = 'training/'
else:
    folder = 'test/'
Dirs['home'] = os.path.expanduser('~/Research/ature2/DRIVE/'+folder)
Dirs['images'] = Dirs['home'] + 'images/'
Dirs['vessels_gt'] = Dirs['home'] + '1st_manual/'
Dirs['masks_orig'] = Dirs['home'] + 'mask/'
Dirs['masks_fixed'] = Dirs['home'] + 'mask_fixed/'
Dirs['masks_eroded'] = Dirs['home'] + 'mask_eroded/'
for folder in Dirs.values():
    try:
        os.makedirs(folder)
    except OSError:
        pass

# Files
Files = {}
Files['images'] = glob.glob(Dirs['images']+'*.tif')
Files['vessels_gt'] = glob.glob(Dirs['vessels_gt']+'*.gif')
Files['masks_orig'] = glob.glob(Dirs['masks_orig']+'*.gif')


##############################################################################
# FUNCTIONS
##############################################################################
def apply_mask(image,mask):
    maskedImage = image.copy()
    for i in range(np.shape(image)[2]):
        maskedImage[:,:,i] = maskedImage[:,:,i]*mask
    return maskedImage


##############################################################################
# MAIN EXECUTION
##############################################################################
if Flags['readImages']:
    Images = {}
    Images['images'] = []
    Images['vessels_gt'] = []
    Images['masks_orig'] = []
    Images['masks_fixed'] = []
    for i in range(len(Files['images'])):
        Images['images'].append(ndimage.imread(Files['images'][i]))
        Images['vessels_gt'].append(ndimage.imread(Files['vessels_gt'][i]))
        Images['masks_orig'].append(ndimage.imread(Files['masks_orig'][i]))


# Fix masks
if Flags['fixMasks']:
    # Tunable parameters
    thresh = 20
    width = 3

    # Find the pixels that are far from zero
    for image in Images['images']:
        fixedMask = image[:,:,1] > thresh
        Images['masks_fixed'].append(fixedMask)

    # Find mask perimeters
    Images['perims'] = []
    h = np.ones((3,3))
    for mask in Images['masks_fixed']:
        perim = signal.convolve2d(mask,h,mode='same')
        perim[perim == 9] = 0
        perim[mask == 0] = 0
        Images['perims'].append(perim)

    # Dilate perimeters
    Images['perims_dil'] = []
    for perim in Images['perims']:
        perim = ndimage.binary_dilation(perim,np.ones((width,width)))
        Images['perims_dil'].append(perim)

    # Erode mask
    Images['masks_eroded'] = []
    for mask,perim in zip(Images['masks_fixed'],Images['perims_dil']):
        erodedMask = mask.copy()
        erodedMask[mask*perim > 0] = 0
        Images['masks_eroded'].append(erodedMask)

    # Save eroded masks
    maskNames = os.listdir(Dirs['masks_orig'])
    for maskFixed,maskEroded,fname in zip(Images['masks_fixed'],
      Images['masks_eroded'],maskNames):
        scipy.misc.imsave(Dirs['masks_fixed'] + fname, maskFixed)
        scipy.misc.imsave(Dirs['masks_eroded'] + fname, maskEroded)

