import torch

import neuralnet.mapnet.main as mapnet
import neuralnet.unet.main as unet

if __name__ == "__main__":
    torch.cuda.set_device(1)
    unet.main()
    mapnet.main()
