import torch
import neuralnet.tracknet.main as tracknet

if __name__ == "__main__":
    torch.cuda.set_device(1)
    tracknet.main()
