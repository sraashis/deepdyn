import torch
import torch.nn.functional as F
from torch import nn

from neuralnet.utils.weights_utils import initialize_weights


class BasicConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, k, s, p):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=False)


class InceptionCrazyNet(nn.Module):
    def __init__(self, num_channels, num_class):
        super(InceptionCrazyNet, self).__init__()

        self.inception1 = BasicConv2d(in_ch=num_channels, out_ch=64, k=3, s=1, p=1)
        self.inception2 = BasicConv2d(in_ch=64, out_ch=128, k=3, s=1, p=1)

        self.inception3 = BasicConv2d(in_ch=num_channels, out_ch=64, k=1, s=1, p=0)
        self.inception4 = BasicConv2d(in_ch=64, out_ch=128, k=1, s=1, p=0)

        self.inception5 = BasicConv2d(in_ch=256, out_ch=128, k=3, s=1, p=1)
        self.inception6 = BasicConv2d(in_ch=128, out_ch=256, k=3, s=1, p=1)

        self.inception7 = BasicConv2d(in_ch=512, out_ch=512, k=3, s=1, p=1)
        self.inception8 = BasicConv2d(in_ch=512, out_ch=256, k=3, s=1, p=1)

        self.inception9 = BasicConv2d(in_ch=512, out_ch=512, k=3, s=1, p=1)
        self.inception10 = BasicConv2d(in_ch=512, out_ch=256, k=3, s=1, p=1)
        self.inception11 = BasicConv2d(in_ch=256, out_ch=128, k=3, s=1, p=1)
        self.inception12 = BasicConv2d(in_ch=128, out_ch=128, k=1, s=1, p=0)
        self.out_conv = nn.Conv2d(in_channels=128, out_channels=num_class, kernel_size=1, stride=1, padding=0)
        initialize_weights(self)

    def forward(self, x):
        x_1 = self.inception1(x)
        x_2 = self.inception2(x_1)

        x_3 = self.inception3(x)
        x_4 = self.inception4(x_3)

        x_5 = self.inception5(torch.cat([x_2[:, :, 6:94, 6:94], x_4[:, :, 6:94, 6:94]], 1))
        x_6 = self.inception6(x_5)

        x_7 = self.inception7(
            torch.cat([x_2[:, :, 12:88, 12:88], x_4[:, :, 12:88, 12:88], x_6[:, :, 6:82, 6:82]],
                      1))
        x_8 = self.inception8(x_7)

        x_9 = self.inception9(
            torch.cat([x_2[:, :, 18:82, 18:82], x_4[:, :, 18:82, 18:82], x_8[:, :, 6:70, 6:70]],
                      1))
        x_10 = self.inception10(x_9)
        x_11 = self.inception11(x_10)
        x_12 = self.inception12(x_11)

        out = self.out_conv(x_12)
        return F.log_softmax(out, dim=1)


m = InceptionCrazyNet(num_channels=9, num_class=2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
