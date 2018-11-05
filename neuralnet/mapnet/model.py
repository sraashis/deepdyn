import torch
import torch.nn.functional as F
from torch import nn


class _DoubleConvolution(nn.Module):
    def __init__(self, in_channels, middle_channel, out_channels, p=0):
        super(_DoubleConvolution, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channel, kernel_size=3, padding=p),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel, out_channels, kernel_size=3, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class BabyUNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(BabyUNet, self).__init__()

        self.A1 = _DoubleConvolution(num_channels, 32, 64, p=1)
        self.A2 = _DoubleConvolution(64, 64, 128, p=1)
        self.Aup = nn.ConvTranspose2d(128, 128, 2, 2)
        self.A3 = _DoubleConvolution(192, 128, 64, p=1)
        self.out = nn.Conv2d(64, num_classes, 1, 1)

    def forward(self, x):
        a1 = self.A1(x)

        a1_up = F.max_pool2d(a1, 2, 2)
        a2 = self.A2(a1_up)
        aup = self.Aup(a2)

        a3 = self.A3(torch.cat([a1, aup], 1))
        out = self.out(a3)

        return F.softmax(out, 1)

    @staticmethod
    def match_and_concat(bypass, upsampled, crop=True):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)


m = BabyUNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
