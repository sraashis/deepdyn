import torch
import torch.nn.functional as F
from torch import nn

from neuralnet.utils.weights_utils import initialize_weights


class _DoubleConvolution(nn.Module):
    def __init__(self, in_channels, middle_channel, out_channels):
        super(_DoubleConvolution, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class UNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(UNet, self).__init__()
        self.dwn1 = _DoubleConvolution(num_channels, 32, 64)
        self.dwn2 = _DoubleConvolution(64, 64, 128)
        self.dwn3 = _DoubleConvolution(128, 128, 256)
        self.middle = _DoubleConvolution(256, 256, 256)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3_out1 = _DoubleConvolution(256, 64, 64)
        self.up3_out2 = _DoubleConvolution(64, 32, 32)
        self.up3_out3 = _DoubleConvolution(32, 8, 8)
        self.up3_fc1 = nn.Linear(8 * 2 * 2, 256)
        self.up3_fc2 = nn.Linear(256, num_classes)

        self.up2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.up2_out1 = _DoubleConvolution(192, 64, 64)
        self.up2_out2 = _DoubleConvolution(64, 32, 32)
        self.up2_out3 = _DoubleConvolution(32, 8, 8)
        self.up2_fc1 = nn.Linear(8 * 4 * 4, 256)
        self.up2_fc2 = nn.Linear(256, num_classes)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1_out1 = _DoubleConvolution(128, 64, 64)
        self.up1_out2 = _DoubleConvolution(64, 32, 32)
        self.up1_out3 = _DoubleConvolution(32, 8, 8)
        self.up1_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.up1_fc2 = nn.Linear(256, num_classes)

        initialize_weights(self)

    def forward(self, x):
        dwn1 = self.dwn1(x)
        dwn1 = F.max_pool2d(dwn1, kernel_size=2, stride=2)

        dwn2 = self.dwn2(dwn1)
        dwn2 = F.max_pool2d(dwn2, kernel_size=2, stride=2)

        dwn3 = self.dwn3(dwn2)
        dwn3 = F.max_pool2d(dwn3, kernel_size=2, stride=2)

        middle = self.middle(dwn3)

        up3 = self.up3(UNet.match_and_concat(dwn3, middle))
        up3_out1 = F.max_pool2d(self.up3_out1(up3), kernel_size=2, stride=2)
        up3_out2 = F.max_pool2d(self.up3_out2(up3_out1), kernel_size=2, stride=2)
        up3_out3 = F.max_pool2d(self.up3_out3(up3_out2), kernel_size=2, stride=2)
        up3_flat = up3_out3.view(-1, 8 * 2 * 2)
        up3_fc1 = F.relu(self.up3_fc1(up3_flat))
        up3_fc2 = self.up3_fc2(up3_fc1)

        up2 = self.up2(UNet.match_and_concat(dwn2, up3))
        up2_out1 = F.max_pool2d(self.up2_out1(up2), kernel_size=2, stride=2)
        up2_out2 = F.max_pool2d(self.up2_out2(up2_out1), kernel_size=2, stride=2)
        up2_out3 = F.max_pool2d(self.up2_out3(up2_out2), kernel_size=2, stride=2)
        up2_flat = up2_out3.view(-1, 8 * 4 * 4)
        up2_fc1 = F.relu(self.up2_fc1(up2_flat))
        up2_fc2 = self.up2_fc2(up2_fc1)

        up1 = self.up1(UNet.match_and_concat(dwn1, up2))
        up1_out1 = F.max_pool2d(self.up1_out1(up1), kernel_size=2, stride=2)
        up1_out2 = F.max_pool2d(self.up1_out2(up1_out1), kernel_size=2, stride=2)
        up1_out3 = F.max_pool2d(self.up1_out3(up1_out2), kernel_size=2, stride=2)
        up1_flat = up1_out3.view(-1, 8 * 8 * 8)
        up1_fc1 = F.relu(self.up1_fc1(up1_flat))
        up1_fc2 = self.up1_fc2(up1_fc1)

        return torch.cat([up3_fc2, up2_fc2, up1_fc2], 1)

    @staticmethod
    def match_and_concat(bypass, upsampled, crop=True):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)


m = UNet(num_channels=2, num_classes=1)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
