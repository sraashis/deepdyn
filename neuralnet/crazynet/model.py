import torch
import torch.nn.functional as F
from torch import nn

from neuralnet.utils.weights_utils import initialize_weights


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

        c = 1
        self.A1_ = _DoubleConvolution(num_channels, int(64 / c), int(64 / c))
        self.A2_ = _DoubleConvolution(int(64 / c), int(128 / c), int(128 / c))
        self.A3_ = _DoubleConvolution(int(128 / c), int(256 / c), int(256 / c))

        self.A_mid = _DoubleConvolution(int(256 / c), int(512 / c), int(512 / c))

        self.A3_up = nn.ConvTranspose2d(int(512 / c), int(256 / c), kernel_size=2, stride=2)
        self._A3 = _DoubleConvolution(int(512 / c), int(256 / c), int(256 / c))

        self.A2_up = nn.ConvTranspose2d(int(256 / c), int(128 / c), kernel_size=2, stride=2)
        self._A2 = _DoubleConvolution((256 / c), int(128 / c), int(128 / c))

        self.A1_up = nn.ConvTranspose2d(int(128 / c), int(64 / c), kernel_size=2, stride=2)
        self._A1 = _DoubleConvolution(int(128 / c), int(64 / c), num_classes)

    def forward(self, x):
        a1_ = self.A1_(x)
        a1_dwn = F.max_pool2d(a1_, kernel_size=2, stride=2)

        a2_ = self.A2_(a1_dwn)
        a2_dwn = F.max_pool2d(a2_, kernel_size=2, stride=2)

        a3_ = self.A3_(a2_dwn)
        a3_dwn = F.max_pool2d(a3_, kernel_size=2, stride=2)

        a_mid = self.A_mid(a3_dwn)

        a3_up = self.A3_up(a_mid)
        _a3 = self._A3(BabyUNet.match_and_concat(a3_, a3_up))

        a2_up = self.A2_up(_a3)
        _a2 = self._A2(BabyUNet.match_and_concat(a2_, a2_up))

        a1_up = self.A1_up(_a2)
        _a1 = self._A1(BabyUNet.match_and_concat(a1_, a1_up))

        return _a1, _a3

    @staticmethod
    def match_and_concat(bypass, upsampled, crop=True):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)


class UUNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(UUNet, self).__init__()

        self.unet0 = BabyUNet(num_channels, 64)
        self.unet1 = BabyUNet(num_channels, 64)
        self.unet2 = BabyUNet(num_channels, 64)
        self.unet3 = BabyUNet(num_channels, 64)
        self.unet4 = BabyUNet(num_channels, 64)
        self.unet5 = BabyUNet(num_channels, 64)
        self.unet6 = BabyUNet(num_channels, 64)
        self.unet7 = BabyUNet(num_channels, 64)
        self.unet8 = BabyUNet(num_channels, 64)

        self.a3_up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.a3 = _DoubleConvolution(256, 128, 128, p=1)

        self.a3_up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.cleaner = _DoubleConvolution(128, 64, 32, p=1)
        self.out = nn.Conv2d(32, num_classes, 1, 1)
        initialize_weights(self)

    def forward(self, x):
        unet0, a3_0 = self.unet0(x[:, 0, :, :].unsqueeze(1))
        unet1, a3_1 = self.unet1(x[:, 1, :, :].unsqueeze(1))
        unet2, a3_2 = self.unet2(x[:, 2, :, :].unsqueeze(1))
        unet3, a3_3 = self.unet3(x[:, 3, :, :].unsqueeze(1))
        unet4, a3_4 = self.unet4(x[:, 4, :, :].unsqueeze(1))
        unet5, a3_5 = self.unet5(x[:, 5, :, :].unsqueeze(1))
        unet6, a3_6 = self.unet6(x[:, 6, :, :].unsqueeze(1))
        unet7, a3_7 = self.unet7(x[:, 7, :, :].unsqueeze(1))
        unet8, a3_8 = self.unet8(x[:, 8, :, :].unsqueeze(1))

        unet_r1 = torch.cat([unet0, unet1, unet2], 3)
        unet_r2 = torch.cat([unet3, unet4, unet5], 3)
        unet_r3 = torch.cat([unet6, unet7, unet8], 3)
        unet = torch.cat([unet_r1, unet_r2, unet_r3], 2)

        a3_r1 = torch.cat([a3_0, a3_1, a3_2], 3)
        a3_r2 = torch.cat([a3_3, a3_4, a3_5], 3)
        a3_r3 = torch.cat([a3_6, a3_7, a3_8], 3)
        a3 = torch.cat([a3_r1, a3_r2, a3_r3], 2)

        a3 = self.a3_up1(a3)
        a3 = self.a3(a3)
        a3 = self.a3_up2(a3)

        clean = self.cleaner(BabyUNet.match_and_concat(a3, unet))
        return F.log_softmax(self.out(clean), 1)


m = UUNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
