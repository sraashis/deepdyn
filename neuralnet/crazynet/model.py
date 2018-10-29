import torch
import torch.nn.functional as F
from torch import nn

from neuralnet.utils.weights_utils import initialize_weights


class _DoubleConvolution(nn.Module):
    def __init__(self, in_channels, middle_channel, out_channels, p=1):
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

        reduce_by = 2

        self.A1_ = _DoubleConvolution(num_channels, int(64 / reduce_by), int(64 / reduce_by))
        self.A2_ = _DoubleConvolution(int(64 / reduce_by), int(128 / reduce_by), int(128 / reduce_by))
        self.A3_ = _DoubleConvolution(int(128 / reduce_by), int(256 / reduce_by), int(256 / reduce_by))
        self.A4_ = _DoubleConvolution(int(256 / reduce_by), int(512 / reduce_by), int(512 / reduce_by))

        self.A_mid = _DoubleConvolution(int(512 / reduce_by), int(1024 / reduce_by), int(1024 / reduce_by))

        self.A4_up = nn.ConvTranspose2d(int(1024 / reduce_by), int(512 / reduce_by), kernel_size=2, stride=2)
        self._A4 = _DoubleConvolution(int(1024 / reduce_by), int(512 / reduce_by), int(512 / reduce_by))

        self.A3_up = nn.ConvTranspose2d(int(512 / reduce_by), int(256 / reduce_by), kernel_size=2, stride=2)
        self._A3 = _DoubleConvolution(int(512 / reduce_by), int(256 / reduce_by), int(256 / reduce_by))

        self.A2_up = nn.ConvTranspose2d(int(256 / reduce_by), int(128 / reduce_by), kernel_size=2, stride=2)
        self._A2 = _DoubleConvolution(int(256 / reduce_by), int(128 / reduce_by), int(128 / reduce_by))

        self.A1_up = nn.ConvTranspose2d(int(128 / reduce_by), int(64 / reduce_by), kernel_size=2, stride=2)
        self._A1 = _DoubleConvolution(int(128 / reduce_by), int(64 / reduce_by), num_classes)

    def forward(self, x):
        a1_ = self.A1_(x)
        a1_dwn = F.max_pool2d(a1_, kernel_size=2, stride=2)

        a2_ = self.A2_(a1_dwn)
        a2_dwn = F.max_pool2d(a2_, kernel_size=2, stride=2)

        a3_ = self.A3_(a2_dwn)
        a3_dwn = F.max_pool2d(a3_, kernel_size=2, stride=2)

        a4_ = self.A4_(a3_dwn)
        a4_dwn = F.max_pool2d(a4_, kernel_size=2, stride=2)

        a_mid = self.A_mid(a4_dwn)

        a4_up = self.A4_up(a_mid)
        _a4 = self._A4(BabyUNet.match_and_concat(a4_, a4_up))

        a3_up = self.A3_up(_a4)
        _a3 = self._A3(BabyUNet.match_and_concat(a3_, a3_up))

        a2_up = self.A2_up(_a3)
        _a2 = self._A2(BabyUNet.match_and_concat(a2_, a2_up))

        a1_up = self.A1_up(_a2)
        _a1 = self._A1(BabyUNet.match_and_concat(a1_, a1_up))

        return a_mid, _a4, _a3, _a2, _a1

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

        self.up0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dc0 = _DoubleConvolution(512, 256, 256)

        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dc1 = _DoubleConvolution(384, 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dc2 = _DoubleConvolution(192, 128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dc3 = _DoubleConvolution(64, 64, 64)

        self.uu_dc1 = _DoubleConvolution(128, 64, 64)
        self.unet = nn.Conv2d(64, num_classes, 1, 1)
        initialize_weights(self)

    def forward(self, x):
        unet0_mid, unet0_a4, unet0_a3, unet0_a2, unet0 = self.unet0(x[:, 0, :, :].unsqueeze(1))
        unet1_mid, unet1_a4, unet1_a3, unet1_a2, unet1 = self.unet1(x[:, 1, :, :].unsqueeze(1))
        unet2_mid, unet2_a4, unet2_a3, unet2_a2, unet2 = self.unet2(x[:, 2, :, :].unsqueeze(1))
        unet3_mid, unet3_a4, unet3_a3, unet3_a2, unet3 = self.unet3(x[:, 3, :, :].unsqueeze(1))

        m1 = torch.cat([unet0_mid, unet1_mid], 3)
        m2 = torch.cat([unet2_mid, unet3_mid], 3)
        mid = torch.cat([m1, m2], 2)

        a4_1 = torch.cat([unet0_a4, unet1_a4], 3)
        a4_2 = torch.cat([unet2_a4, unet3_a4], 3)
        a4 = torch.cat([a4_1, a4_2], 2)
        mid = self.dc0(torch.cat([a4, self.up0(mid)], 1))

        a3_1 = torch.cat([unet0_a3, unet1_a3], 3)
        a3_2 = torch.cat([unet2_a3, unet3_a3], 3)
        a3 = torch.cat([a3_1, a3_2], 2)
        mid = self.dc1(torch.cat([a3, self.up1(mid)], 1))

        a2_1 = torch.cat([unet0_a2, unet1_a2], 3)
        a2_2 = torch.cat([unet2_a2, unet3_a2], 3)
        a2 = torch.cat([a2_1, a2_2], 2)
        mid = self.dc2(torch.cat([a2, self.up2(mid)], 1))

        mid = self.dc3(self.up3(mid))

        r1 = torch.cat([unet0, unet1], 3)
        r2 = torch.cat([unet2, unet3], 3)
        unet = torch.cat([r1, r2], 2)

        all = torch.cat([mid, unet], 1)
        all = F.dropout2d(all, 0.4)

        all = self.uu_dc1(all)
        return F.log_softmax(self.unet(all), 1)


m = UUNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
