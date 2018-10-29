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

        self.A1_ = _DoubleConvolution(num_channels, 64, 64)
        self.A2_ = _DoubleConvolution(64, 128, 128)
        self.A3_ = _DoubleConvolution(128, 256, 256)

        self.A_mid = _DoubleConvolution(256, 512, 512)

        self.A3_up = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self._A3 = _DoubleConvolution(512, 256, 256)

        self.A2_up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self._A2 = _DoubleConvolution(256, 128, 128)

        self.A1_up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self._A1 = _DoubleConvolution(128, 64, num_classes)

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

        return [a_mid, _a3, _a2, _a1]

    @staticmethod
    def match_and_concat(bypass, upsampled, crop=True):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)


class UUNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(UUNet, self).__init__()
        unets = []
        for i in range(9):
            unets.append(BabyUNet(num_channels, 16))

        self.unets = nn.Sequential(*unets)

        self.a3_1up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.a3_1 = _DoubleConvolution(128, 64, 64)

        self.a3_2up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.a3_2 = _DoubleConvolution(32, 16, 16)

        self.out = nn.Conv2d(32, num_classes, 3, 1, 1)
        initialize_weights(self)

    def forward(self, x):

        baby_unets = []
        for i in range(9):
            baby_unets.append(self.unets[i](x[:, i, :, :].unsqueeze(1)))

        # mid_r1 = torch.cat([baby_unets[0][0], baby_unets[1][0], baby_unets[2][0]], 3)
        # mid_r2 = torch.cat([baby_unets[3][0], baby_unets[4][0], baby_unets[5][0]], 3)
        # mid_r3 = torch.cat([baby_unets[6][0], baby_unets[7][0], baby_unets[8][0]], 3)
        # mid = torch.cat([mid_r1, mid_r2, mid_r3], 2)

        a3_r1 = torch.cat([baby_unets[0][1], baby_unets[1][1], baby_unets[2][1]], 3)
        a3_r2 = torch.cat([baby_unets[3][1], baby_unets[4][1], baby_unets[5][1]], 3)
        a3_r3 = torch.cat([baby_unets[6][1], baby_unets[7][1], baby_unets[8][1]], 3)
        a3 = torch.cat([a3_r1, a3_r2, a3_r3], 2)

        a3 = self.a3_1(self.a3_1up(a3))
        a3 = self.a3_2(self.a3_2up(a3))

        # a2_r1 = torch.cat([baby_unets[0][2], baby_unets[1][2], baby_unets[2][2]], 3)
        # a2_r2 = torch.cat([baby_unets[3][2], baby_unets[4][2], baby_unets[5][2]], 3)
        # a2_r3 = torch.cat([baby_unets[6][2], baby_unets[7][2], baby_unets[8][2]], 3)
        # a2 = torch.cat([a2_r1, a2_r2, a2_r3], 2)

        unet_r1 = torch.cat([baby_unets[0][3], baby_unets[1][3], baby_unets[2][3]], 3)
        unet_r2 = torch.cat([baby_unets[3][3], baby_unets[4][3], baby_unets[5][3]], 3)
        unet_r3 = torch.cat([baby_unets[6][3], baby_unets[7][3], baby_unets[8][3]], 3)
        unet = torch.cat([unet_r1, unet_r2, unet_r3], 2)

        return F.log_softmax(self.out(BabyUNet.match_and_concat(a3, unet)), 1)


m = UUNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
