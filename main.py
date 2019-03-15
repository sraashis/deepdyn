import testarch.unet as net
import testarch.unet.runs as r
import torchvision.transforms as tmf

import torch
torch.cuda.set_device(1)

transforms = tmf.Compose([
    tmf.ToPILImage(),
    tmf.ToTensor()
])

runs = [r.DRIVE]
if __name__ == "__main__":
    net.run(runs, transforms)

