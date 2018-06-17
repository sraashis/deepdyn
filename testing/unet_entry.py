# !/home/akhanal1/miniconda3//bin/python3.5
# Torch imports
import os
import sys

from testing.unet_runner import UnetRunner

sys.path.append('/home/ak/PycharmProjects/ature')
os.chdir('/home/ak/PycharmProjects/ature')

import torchvision.transforms as transforms

if __name__ == "__main__":
    sep = os.sep
    Params = {}
    Params['num_channels'] = 1
    Params['classes'] = {'background': 0, 'vessel': 1, }
    Params['batch_size'] = 1
    Params['num_classes'] = len(Params['classes'])
    Params['epochs'] = 1000
    Params['patch_size'] = (388, 388)  # rows X cols
    Params['use_gpu'] = True
    Params['learning_rate'] = 0.0001

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    runner = UnetRunner(Params=Params,
                        transform=transform)

    ##################### DRIVE DATASET ########################
    Dirs = {}
    Dirs['data'] = 'data' + sep + 'DRIVE' + sep + 'training'
    Dirs['images'] = Dirs['data'] + sep + 'images'
    Dirs['mask'] = Dirs['data'] + sep + 'mask'
    Dirs['truth'] = Dirs['data'] + sep + '1st_manual'

    TestDirs = {}
    TestDirs['data'] = 'data' + sep + 'DRIVE' + sep + 'testing'
    TestDirs['images'] = TestDirs['data'] + sep + 'images'
    TestDirs['mask'] = TestDirs['data'] + sep + 'mask'
    TestDirs['truth'] = TestDirs['data'] + sep + '1st_manual'
    TestDirs['segmented'] = TestDirs['data'] + sep + 'segmented'

    ValidationDirs = {}
    ValidationDirs['data'] = 'data' + sep + 'DRIVE' + sep + 'testing'
    ValidationDirs['images'] = ValidationDirs['data'] + sep + 'validation_images'
    ValidationDirs['mask'] = ValidationDirs['data'] + sep + 'mask'
    ValidationDirs['truth'] = ValidationDirs['data'] + sep + '1st_manual'

    for k, folder in Dirs.items():
        os.makedirs(folder, exist_ok=True)
    for k, folder in TestDirs.items():
        os.makedirs(folder, exist_ok=True)
    for k, folder in ValidationDirs.items():
        os.makedirs(folder, exist_ok=True)


    def get_mask_file(file_name):
        return file_name.split('_')[0] + '_training_mask.gif'


    def get_ground_truth_file(file_name):
        return file_name.split('_')[0] + '_manual1.gif'


    def get_mask_file_test(file_name):
        return file_name.split('_')[0] + '_test_mask.gif'


    checkpoint_file = 'chkDRIVEunet.tar'
    runner.train(Dirs=Dirs, ValidationDirs=ValidationDirs,
                 train_mask_getter=get_mask_file, train_groundtruth_getter=get_ground_truth_file,
                 val_mask_getter=get_mask_file_test, val_groundtruth_getter=get_ground_truth_file,
                 checkpoint_file=checkpoint_file)

    runner.run_tests(TestDirs=TestDirs,
                     test_mask_getter=get_mask_file_test,
                     test_groundtruth_file_getter=get_ground_truth_file, checkpoint_file=checkpoint_file)
    #################################################################################

    ############## AV-WIDE Dataset ##################################################
    Dirs = {}
    Dirs['checkpoint'] = 'assests' + sep + 'nnet_models'
    Dirs['data'] = 'data' + sep + 'AV-WIDE' + sep + 'training'
    Dirs['images'] = Dirs['data'] + sep + 'images'
    Dirs['mask'] = Dirs['data'] + sep + 'mask'
    Dirs['truth'] = Dirs['data'] + sep + '1st_manual'

    TestDirs = {}
    TestDirs['data'] = 'data' + sep + 'AV-WIDE' + sep + 'testing'
    TestDirs['images'] = TestDirs['data'] + sep + 'images'
    TestDirs['mask'] = TestDirs['data'] + sep + 'mask'
    TestDirs['truth'] = TestDirs['data'] + sep + '1st_manual'
    TestDirs['segmented'] = TestDirs['data'] + sep + 'segmented'

    ValidationDirs = {}
    ValidationDirs['data'] = 'data' + sep + 'AV-WIDE' + sep + 'testing'
    ValidationDirs['images'] = ValidationDirs['data'] + sep + 'validation_images'
    ValidationDirs['mask'] = ValidationDirs['data'] + sep + 'mask'
    ValidationDirs['truth'] = ValidationDirs['data'] + sep + '1st_manual'

    for k, folder in Dirs.items():
        os.makedirs(folder, exist_ok=True)
    for k, folder in TestDirs.items():
        os.makedirs(folder, exist_ok=True)
    for k, folder in ValidationDirs.items():
        os.makedirs(folder, exist_ok=True)


    def get_mask_file(file_name):
        return file_name.split('_')[0] + '_training_mask.gif'


    def get_ground_truth_file(file_name):
        return file_name.split('.')[0] + '_vessels.png'

    checkpoint_file = 'chkWIDEunet.tar'
    runner.train(Dirs=Dirs, ValidationDirs=ValidationDirs,
                 train_mask_getter=get_mask_file, train_groundtruth_getter=get_ground_truth_file,
                 val_mask_getter=get_mask_file_test, val_groundtruth_getter=get_ground_truth_file,
                 checkpoint_file=checkpoint_file)

    runner.run_tests(TestDirs=TestDirs,
                     test_mask_getter=get_mask_file_test,
                     test_groundtruth_file_getter=get_ground_truth_file, checkpoint_file=checkpoint_file)
    ######################################################################################
