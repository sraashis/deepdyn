import torch
import torch.nn.functional as F
from torch import nn

from neuralnet.utils.weights_utils import initialize_weights


class BasicConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, k, s, p):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        # self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return F.relu(x, inplace=False)


class InceptionThrNet(nn.Module):
    def __init__(self, num_channels, num_class):
        super(InceptionThrNet, self).__init__()

        self.inception1 = BasicConv2d(in_ch=num_channels, out_ch=64, k=3, s=1, p=1)
        self.inception2 = BasicConv2d(in_ch=64, out_ch=128, k=3, s=1, p=1)
        self.inception3 = BasicConv2d(in_ch=128, out_ch=128, k=3, s=1, p=1)
        self.inception4 = BasicConv2d(in_ch=128, out_ch=256, k=3, s=1, p=1)

        self.inception5 = BasicConv2d(in_ch=256, out_ch=256, k=3, s=1, p=1)
        self.inception6 = BasicConv2d(in_ch=256, out_ch=256, k=3, s=1, p=1)
        self.inception7 = BasicConv2d(in_ch=256, out_ch=256, k=3, s=1, p=1)
        self.inception8 = BasicConv2d(in_ch=256, out_ch=256, k=3, s=1, p=1)

        self.inception9 = BasicConv2d(in_ch=512, out_ch=128, k=3, s=1, p=1)
        self.inception10 = BasicConv2d(in_ch=128, out_ch=128, k=3, s=1, p=1)
        self.inception11 = BasicConv2d(in_ch=128, out_ch=128, k=3, s=1, p=1)
        self.inception12 = BasicConv2d(in_ch=128, out_ch=64, k=3, s=1, p=1)

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_class)
        initialize_weights(self)

    def forward(self, x):
        x_1 = self.inception1(x)
        x_2 = self.inception2(x_1)
        x_3 = self.inception3(x_2)
        x_4 = self.inception4(x_3)
        x_4 = F.max_pool2d(x_4, kernel_size=2, stride=2, padding=0)

        x_5 = self.inception5(x_4)
        x_6 = self.inception6(x_5)
        x_7 = self.inception7(x_6)
        x_8 = self.inception8(x_7)
        x_8 = F.max_pool2d(torch.cat([x_8, x_4], 1), kernel_size=2, stride=2, padding=0)

        x_9 = self.inception9(x_8)
        x_10 = self.inception10(x_9)
        x_11 = self.inception11(x_10)
        x_12 = self.inception12(x_11)
        x_12 = F.max_pool2d(x_12, kernel_size=2, stride=2, padding=0)

        flat = x_12.view(-1, 64 * 4 * 4)
        flat = F.relu(self.fc1(flat))
        return self.fc2(flat)
        # return 0.1 * (torch.abs(128 - self.fc2(flat)))


m = InceptionThrNet(num_channels=4, num_class=1)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
