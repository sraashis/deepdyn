import os

import PIL.Image as IMG
import torch
import torch.optim as optim
from torch.utils.data.dataset import Dataset

import neuralnet.unet.utils as ut
from commons.IMAGE import SegmentedImage
from neuralnet.unet.model.unet import UNet
from neuralnet.unet.unet_dataloader import PatchesGenerator, PatchesGeneratorPerImgObj
from neuralnet.unet.unet_trainer import UNetNNTrainer

sep = os.sep


def train(Params, Dirs, ValidationDirs, transform, train_mask_getter, train_groundtruth_getter,
          val_mask_getter,
          val_groundtruth_getter, check_point_file):
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # """
    # ### Load train data
    trainset = PatchesGenerator(Dirs=Dirs, train_image_size=Params['patch_size'],
                                transform=transform,
                                fget_mask=train_mask_getter,
                                fget_truth=train_groundtruth_getter)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=Params['batch_size'], shuffle=False, num_workers=3)

    # ### Load Validation Data
    validation_set = PatchesGenerator(Dirs=ValidationDirs, train_image_size=Params['patch_size'],
                                      transform=transform,
                                      fget_mask=val_mask_getter,
                                      fget_truth=val_groundtruth_getter)

    validationloader = torch.utils.data.DataLoader(validation_set, batch_size=Params['batch_size'], shuffle=False,
                                                   num_workers=3)

    # """
    # ### Define the network
    net = UNet(Params['num_channels'], Params['num_classes'])
    optimizer = optim.Adam(net.parameters(), lr=Params['learning_rate'])

    # ### Train and evaluate network
    trainer = UNetNNTrainer(model=net, checkpoint_dir=Params['checkpoint_dir'],
                            checkpoint_file='TRAIN-' + check_point_file,
                            log_to_file=True,
                            use_gpu=Params['use_gpu'])
    trainer.train(optimizer=optimizer, dataloader=trainloader, epochs=Params['epochs'],
                  validationloader=validationloader, force_checkpoint=False, log_frequency=20)

    with open(os.path.join(Params['checkpoint_dir']) + sep + 'PARAMS-' + check_point_file + '.txt') as pfile:
        pfile.write(Params)


def run_tests(TestDirs, Params, transform, test_mask_getter, test_groundtruth_file_getter, checkpoint_file):
    # ### Define the network
    net = UNet(Params['num_channels'], Params['num_classes'])
    trainer = UNetNNTrainer(model=net, checkpoint_dir=Params['checkpoint_dir'],
                            checkpoint_file='TEST-' + checkpoint_file,
                            log_to_file=True,
                            use_gpu=Params['use_gpu'])
    trainer.resume_from_checkpoint()
    for filename in os.listdir(TestDirs['images']):
        img_obj = SegmentedImage()
        img_obj.load_file(data_dir=TestDirs['images'], file_name=filename)
        img_obj.load_mask(mask_dir=TestDirs['mask'], fget_mask=test_mask_getter, erode=True)
        img_obj.load_ground_truth(gt_dir=TestDirs['truth'], fget_ground_truth=test_groundtruth_file_getter)

        testset = PatchesGeneratorPerImgObj(img_obj=img_obj, train_image_size=Params['patch_size'],
                                            transform=transform)

        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                 shuffle=False, num_workers=0, sampler=None)
        scores, y_pred, y_true = trainer.evaluate(dataloader=testloader, force_checkpoint=False)
        ppp = ut.merge_patches(scores, img_obj.working_arr.shape, Params['patch_size'])
        IMG.fromarray(ppp).save(TestDirs['segmented'] + sep + filename + '.png')

# ### FAST MST algorithm
# params = {'sk_threshold': 150,
#           'alpha': 7.0,
#           'orig_contrib': 0.3,
#           'seg_threshold': 24}
#
# img_obj.working_arr = None  # todo
# img_obj.generate_skeleton(threshold=params['sk_threshold'])
# img_obj.generate_lattice_graph()
