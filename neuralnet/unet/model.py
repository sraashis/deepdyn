import torch
import torch.nn.functional as F
from torch import nn

from neuralnet.utils.weights_utils import initialize_weights


class _DoubleConvolution(nn.Module):
    def __init__(self, in_channels, middle_channel, out_channels):
        super(_DoubleConvolution, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channel, kernel_size=3),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class UNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(UNet, self).__init__()
        mul = 16 if torch.cuda.is_available() else 1
        self.enc1 = _DoubleConvolution(num_channels, 4 * mul, 4 * mul)
        self.enc2 = _DoubleConvolution(4 * mul, 8 * mul, 8 * mul)
        self.enc3 = _DoubleConvolution(8 * mul, 16 * mul, 16 * mul)
        self.enc4 = _DoubleConvolution(16 * mul, 32 * mul, 32 * mul)

        self.dec4 = _DoubleConvolution(32 * mul, 64 * mul, 64 * mul)
        self.dec4_up = nn.ConvTranspose2d(64 * mul, 32 * mul, kernel_size=2, stride=2)

        self.dec3 = _DoubleConvolution(64 * mul, 32 * mul, 32 * mul)
        self.dec3_up = nn.ConvTranspose2d(32 * mul, 16 * mul, kernel_size=2, stride=2)

        self.dec2 = _DoubleConvolution(32 * mul, 16 * mul, 16 * mul)
        self.dec2_up = nn.ConvTranspose2d(16 * mul, 8 * mul, kernel_size=2, stride=2)

        self.dec1 = _DoubleConvolution(16 * mul, 8 * mul, 8 * mul)
        self.dec1_up = nn.ConvTranspose2d(8 * mul, 4 * mul, kernel_size=2, stride=2)

        self.out = _DoubleConvolution(8 * mul, 4 * mul, 4 * mul)
        self.final = nn.Conv2d(4 * mul, num_classes, kernel_size=1)

        initialize_weights(self)

    def forward(self, x):
        enc1_ = self.enc1(x)
        enc1 = F.max_pool2d(enc1_, kernel_size=2, stride=2)

        enc2_ = self.enc2(enc1)
        enc2 = F.max_pool2d(enc2_, kernel_size=2, stride=2)

        enc3_ = self.enc3(enc2)
        enc3 = F.max_pool2d(enc3_, kernel_size=2, stride=2)

        enc4_ = self.enc4(enc3)
        enc4 = F.max_pool2d(enc4_, kernel_size=2, stride=2)
        enc4 = F.dropout2d(enc4, p=0.2)

        dec4 = self.dec4(enc4)
        dec4 = F.dropout2d(dec4, p=0.5)

        dec3 = self.dec3(UNet.match_and_concat(enc4_, self.dec4_up(dec4)))
        dec3 = F.dropout2d(dec3, p=0.2)

        dec2 = self.dec2(UNet.match_and_concat(enc3_, self.dec3_up(dec3)))
        dec1 = self.dec1(UNet.match_and_concat(enc2_, self.dec2_up(dec2)))
        out = self.out(UNet.match_and_concat(enc1_, self.dec1_up(dec1)))
        final = self.final(out)
        return F.log_softmax(final, dim=1)

    @staticmethod
    def match_and_concat(bypass, upsampled, crop=True):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
