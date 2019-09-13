import torchvision.transforms as tmf

import testarch.miniunet.runs as rm
import testarch.unet as unet
import testarch.miniunet as miniunet
import testarch.unet.runs as ru

transforms = tmf.Compose([
    tmf.ToPILImage(),
    tmf.ToTensor()
])

if __name__ == "__main__":
    runs_unet = [ru.DRIVE_1_100_1, ru.DRIVE_1_1, ru.DRIVE_WEIGHTED,
                 ru.STARE_1_100_1, ru.STARE_1_1, ru.STARE_WEIGHTED,
                 ru.WIDE_1_100_1, ru.WIDE_1_1, ru.WIDE_WEIGHTED,
                 ru.CHASEDB_1_100_1, ru.CHASEDB_1_1, ru.CHASEDB_WEIGHTED,
                 ru.VEVIO_MOSAICS_1_100_1, ru.VEVIO_MOSAICS_1_1, ru.VEVIO_MOSAICS_WEIGHTED,
                 ru.VEVIO_FRAMES_1_100_1, ru.VEVIO_FRAMES_1_1, ru.VEVIO_FRAMES_WEIGHTED]

    runs_miniunet = [rm.DRIVE_1_100_1, rm.DRIVE_1_1, rm.DRIVE_WEIGHTED,
                     rm.STARE_1_100_1, rm.STARE_1_1, rm.STARE_WEIGHTED,
                     rm.WIDE_1_100_1, rm.WIDE_1_1, rm.WIDE_WEIGHTED,
                     rm.CHASEDB_1_100_1, rm.CHASEDB_1_1, rm.CHASEDB_WEIGHTED,
                     rm.VEVIO_MOSAICS_1_100_1, rm.VEVIO_MOSAICS_1_1, rm.VEVIO_MOSAICS_WEIGHTED,
                     rm.VEVIO_FRAMES_1_100_1, rm.VEVIO_FRAMES_1_1, rm.VEVIO_FRAMES_WEIGHTED]

    for runet, rminiunet in zip(runs_unet, runs_miniunet):
        unet.run([runet], transforms)
        miniunet.run([rminiunet], transforms)
